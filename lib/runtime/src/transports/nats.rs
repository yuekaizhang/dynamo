// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! NATS transport
//!
//! The following environment variables are used to configure the NATS client:
//!
//! - `NATS_SERVER`: the NATS server address
//!
//! For authentication, the following environment variables are used and prioritized in the following order:
//!
//! - `NATS_AUTH_USERNAME`: the username for authentication
//! - `NATS_AUTH_PASSWORD`: the password for authentication
//! - `NATS_AUTH_TOKEN`: the token for authentication
//! - `NATS_AUTH_NKEY`: the nkey for authentication
//! - `NATS_AUTH_CREDENTIALS_FILE`: the path to the credentials file
//!
//! Note: `NATS_AUTH_USERNAME` and `NATS_AUTH_PASSWORD` must be used together.
use crate::{metrics::MetricsRegistry, Result};

use async_nats::connection::State;
use async_nats::{client, jetstream, Subscriber};
use bytes::Bytes;
use derive_builder::Builder;
use futures::{StreamExt, TryStreamExt};
use prometheus::{Counter, Gauge, Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncRead;
use tokio::time;
use url::Url;
use validator::{Validate, ValidationError};

use crate::metrics::prometheus_names::nats as nats_metrics;
pub use crate::slug::Slug;
use tracing as log;

use super::utils::build_in_runtime;

pub const URL_PREFIX: &str = "nats://";

#[derive(Clone)]
pub struct Client {
    client: client::Client,
    js_ctx: jetstream::Context,
}

impl Client {
    /// Create a NATS [`ClientOptionsBuilder`].
    pub fn builder() -> ClientOptionsBuilder {
        ClientOptionsBuilder::default()
    }

    /// Returns a reference to the underlying [`async_nats::client::Client`] instance
    pub fn client(&self) -> &client::Client {
        &self.client
    }

    /// Returns a reference to the underlying [`async_nats::jetstream::Context`] instance
    pub fn jetstream(&self) -> &jetstream::Context {
        &self.js_ctx
    }

    /// host:port of NATS
    pub fn addr(&self) -> String {
        let info = self.client.server_info();
        format!("{}:{}", info.host, info.port)
    }

    /// fetch the list of streams
    pub async fn list_streams(&self) -> Result<Vec<String>> {
        let names = self.js_ctx.stream_names();
        let stream_names: Vec<String> = names.try_collect().await?;
        Ok(stream_names)
    }

    /// fetch the list of consumers for a given stream
    pub async fn list_consumers(&self, stream_name: &str) -> Result<Vec<String>> {
        let stream = self.js_ctx.get_stream(stream_name).await?;
        let consumers: Vec<String> = stream.consumer_names().try_collect().await?;
        Ok(consumers)
    }

    pub async fn stream_info(&self, stream_name: &str) -> Result<jetstream::stream::State> {
        let mut stream = self.js_ctx.get_stream(stream_name).await?;
        let info = stream.info().await?;
        Ok(info.state.clone())
    }

    pub async fn get_stream(&self, name: &str) -> Result<jetstream::stream::Stream> {
        let stream = self.js_ctx.get_stream(name).await?;
        Ok(stream)
    }

    /// Issues a broadcast request for all services with the provided `service_name` to report their
    /// current stats. Each service will only respond once. The service may have customized the reply
    /// so the caller should select which endpoint and what concrete data model should be used to
    /// extract the details.
    ///
    /// Note: Because each endpoint will only reply once, the caller must drop the subscription after
    /// some time or it will await forever.
    pub async fn scrape_service(&self, service_name: &str) -> Result<Subscriber> {
        let subject = format!("$SRV.STATS.{}", service_name);
        let reply_subject = format!("_INBOX.{}", nuid::next());
        let subscription = self.client.subscribe(reply_subject.clone()).await?;

        // Publish the request with the reply-to subject
        self.client
            .publish_with_reply(subject, reply_subject, "".into())
            .await?;

        Ok(subscription)
    }

    /// Upload file to NATS at this URL
    pub async fn object_store_upload(&self, filepath: &Path, nats_url: Url) -> anyhow::Result<()> {
        let mut disk_file = TokioFile::open(filepath).await?;

        let (bucket_name, key) = url_to_bucket_and_key(&nats_url)?;
        let context = self.jetstream();

        let bucket = match context.get_object_store(&bucket_name).await {
            Ok(bucket) => bucket,
            Err(err) if err.to_string().contains("stream not found") => {
                // err.source() is GetStreamError, which has a kind() which
                // is GetStreamErrorKind::JetStream which wraps a jetstream::Error
                // which has code 404. Phew. So yeah check the string for now.

                tracing::debug!("Creating NATS bucket {bucket_name}");
                context
                    .create_object_store(jetstream::object_store::Config {
                        bucket: bucket_name.to_string(),
                        ..Default::default()
                    })
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed creating bucket / object store: {e}"))?
            }
            Err(err) => {
                anyhow::bail!("NATS get_object_store error: {err}");
            }
        };

        let key_meta = async_nats::jetstream::object_store::ObjectMetadata {
            name: key.to_string(),
            ..Default::default()
        };
        bucket.put(key_meta, &mut disk_file).await.map_err(|e| {
            anyhow::anyhow!("Failed uploading to bucket / object store {bucket_name}/{key}: {e}")
        })?;

        Ok(())
    }

    /// Download file from NATS at this URL
    pub async fn object_store_download(
        &self,
        nats_url: Url,
        filepath: &Path,
    ) -> anyhow::Result<()> {
        let mut disk_file = TokioFile::create(filepath).await?;

        let (bucket_name, key) = url_to_bucket_and_key(&nats_url)?;
        let context = self.jetstream();

        let bucket = match context.get_object_store(&bucket_name).await {
            Ok(bucket) => bucket,
            Err(err) if err.to_string().contains("stream not found") => {
                // err.source() is GetStreamError, which has a kind() which
                // is GetStreamErrorKind::JetStream which wraps a jetstream::Error
                // which has code 404. Phew. So yeah check the string for now.
                anyhow::bail!("NATS get_object_store bucket does not exist: {bucket_name}. {err}.");
            }
            Err(err) => {
                anyhow::bail!("NATS get_object_store error: {err}");
            }
        };

        let mut obj_reader = bucket.get(&key).await.map_err(|e| {
            anyhow::anyhow!(
                "Failed downloading from bucket / object store {bucket_name}/{key}: {e}"
            )
        })?;
        let _bytes_copied = tokio::io::copy(&mut obj_reader, &mut disk_file).await?;

        Ok(())
    }

    /// Delete a bucket and all it's contents from the NATS object store
    pub async fn object_store_delete_bucket(&self, bucket_name: &str) -> anyhow::Result<()> {
        let context = self.jetstream();
        match context.delete_object_store(&bucket_name).await {
            Ok(_) => Ok(()),
            Err(err) if err.to_string().contains("stream not found") => {
                tracing::trace!(bucket_name, "NATS bucket already gone");
                Ok(())
            }
            Err(err) => Err(anyhow::anyhow!("NATS get_object_store error: {err}")),
        }
    }
}

/// NATS client options
///
/// This object uses the builder pattern with default values that are evaluates
/// from the environment variables if they are not explicitly set by the builder.
#[derive(Debug, Clone, Builder, Validate)]
pub struct ClientOptions {
    #[builder(setter(into), default = "default_server()")]
    #[validate(custom(function = "validate_nats_server"))]
    server: String,

    #[builder(default)]
    auth: NatsAuth,
}

fn default_server() -> String {
    if let Ok(server) = std::env::var("NATS_SERVER") {
        return server;
    }

    "nats://localhost:4222".to_string()
}

fn validate_nats_server(server: &str) -> Result<(), ValidationError> {
    if server.starts_with("nats://") {
        Ok(())
    } else {
        Err(ValidationError::new("server must start with 'nats://'"))
    }
}

// TODO(jthomson04): We really shouldn't be hardcoding this.
const NATS_WORKER_THREADS: usize = 4;

impl ClientOptions {
    /// Create a new [`ClientOptionsBuilder`]
    pub fn builder() -> ClientOptionsBuilder {
        ClientOptionsBuilder::default()
    }

    /// Validate the config and attempt to connection to the NATS server
    pub async fn connect(self) -> Result<Client> {
        self.validate()?;

        let client = match self.auth {
            NatsAuth::UserPass(username, password) => {
                async_nats::ConnectOptions::with_user_and_password(username, password)
            }
            NatsAuth::Token(token) => async_nats::ConnectOptions::with_token(token),
            NatsAuth::NKey(nkey) => async_nats::ConnectOptions::with_nkey(nkey),
            NatsAuth::CredentialsFile(path) => {
                async_nats::ConnectOptions::with_credentials_file(path).await?
            }
        };

        let (client, _) = build_in_runtime(
            async move {
                client
                    .connect(self.server)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to connect to NATS: {e}"))
            },
            NATS_WORKER_THREADS,
        )
        .await?;

        let js_ctx = jetstream::new(client.clone());

        Ok(Client { client, js_ctx })
    }
}

impl Default for ClientOptions {
    fn default() -> Self {
        ClientOptions {
            server: default_server(),
            auth: NatsAuth::default(),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum NatsAuth {
    UserPass(String, String),
    Token(String),
    NKey(String),
    CredentialsFile(PathBuf),
}

impl std::fmt::Debug for NatsAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NatsAuth::UserPass(user, _pass) => {
                write!(f, "UserPass({}, <redacted>)", user)
            }
            NatsAuth::Token(_token) => write!(f, "Token(<redacted>)"),
            NatsAuth::NKey(_nkey) => write!(f, "NKey(<redacted>)"),
            NatsAuth::CredentialsFile(path) => write!(f, "CredentialsFile({:?})", path),
        }
    }
}

impl Default for NatsAuth {
    fn default() -> Self {
        if let (Ok(username), Ok(password)) = (
            std::env::var("NATS_AUTH_USERNAME"),
            std::env::var("NATS_AUTH_PASSWORD"),
        ) {
            return NatsAuth::UserPass(username, password);
        }

        if let Ok(token) = std::env::var("NATS_AUTH_TOKEN") {
            return NatsAuth::Token(token);
        }

        if let Ok(nkey) = std::env::var("NATS_AUTH_NKEY") {
            return NatsAuth::NKey(nkey);
        }

        if let Ok(path) = std::env::var("NATS_AUTH_CREDENTIALS_FILE") {
            return NatsAuth::CredentialsFile(PathBuf::from(path));
        }

        NatsAuth::UserPass("user".to_string(), "user".to_string())
    }
}

/// Is this file name / url in the NATS object store?
/// Checks the name only, does not go to the store.
pub fn is_nats_url(s: &str) -> bool {
    s.starts_with(URL_PREFIX)
}

/// Extract NATS bucket and key from a nats URL of the form:
/// nats://host[:port]/bucket/key
pub fn url_to_bucket_and_key(url: &Url) -> anyhow::Result<(String, String)> {
    let Some(mut path_segments) = url.path_segments() else {
        anyhow::bail!("No path in NATS URL: {url}");
    };
    let Some(bucket) = path_segments.next() else {
        anyhow::bail!("No bucket in NATS URL: {url}");
    };
    let Some(key) = path_segments.next() else {
        anyhow::bail!("No key in NATS URL: {url}");
    };
    Ok((bucket.to_string(), key.to_string()))
}

/// A queue implementation using NATS JetStream
pub struct NatsQueue {
    /// The name of the stream to use for the queue
    stream_name: String,
    /// The NATS server URL
    nats_server: String,
    /// Timeout for dequeue operations in seconds
    dequeue_timeout: time::Duration,
    /// The NATS client
    client: Option<Client>,
    /// The subject pattern used for this queue
    subject: String,
    /// The subscriber for pull-based consumption
    subscriber: Option<jetstream::consumer::PullConsumer>,
}

impl NatsQueue {
    /// Create a new NatsQueue with the given configuration
    pub fn new(stream_name: String, nats_server: String, dequeue_timeout: time::Duration) -> Self {
        // Sanitize stream name to remove path separators (like in Python version)
        let sanitized_stream_name = stream_name.replace(['/', '\\'], "_");

        let subject = format!("{}.*", sanitized_stream_name);

        Self {
            stream_name: sanitized_stream_name,
            nats_server,
            dequeue_timeout,
            client: None,
            subject,
            subscriber: None,
        }
    }

    /// Connect to the NATS server and set up the stream and consumer
    pub async fn connect(&mut self) -> Result<()> {
        if self.client.is_none() {
            // Create a new client
            let client_options = Client::builder().server(self.nats_server.clone()).build()?;

            let client = client_options.connect().await?;

            // Check if stream exists, if not create it
            let streams = client.list_streams().await?;
            if !streams.contains(&self.stream_name) {
                log::debug!("Creating NATS stream {}", self.stream_name);
                let stream_config = jetstream::stream::Config {
                    name: self.stream_name.clone(),
                    subjects: vec![self.subject.clone()],
                    max_age: time::Duration::from_secs(60 * 10), // 10 min
                    ..Default::default()
                };
                client.jetstream().create_stream(stream_config).await?;
            }

            // Create persistent subscriber
            let consumer_config = jetstream::consumer::pull::Config {
                durable_name: Some("worker-group".to_string()),
                ..Default::default()
            };

            let stream = client.jetstream().get_stream(&self.stream_name).await?;
            let subscriber = stream.create_consumer(consumer_config).await?;

            self.subscriber = Some(subscriber);
            self.client = Some(client);
        }

        Ok(())
    }

    /// Ensure we have an active connection
    pub async fn ensure_connection(&mut self) -> Result<()> {
        if self.client.is_none() {
            self.connect().await?;
        }
        Ok(())
    }

    /// Close the connection when done
    pub async fn close(&mut self) -> Result<()> {
        self.subscriber = None;
        self.client = None;
        Ok(())
    }

    /// Enqueue a task using the provided data
    pub async fn enqueue_task(&mut self, task_data: Bytes) -> Result<()> {
        self.ensure_connection().await?;

        if let Some(client) = &self.client {
            let subject = format!("{}.queue", self.stream_name);
            client.jetstream().publish(subject, task_data).await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Client not connected"))
        }
    }

    /// Dequeue and return a task as raw bytes
    pub async fn dequeue_task(&mut self, timeout: Option<time::Duration>) -> Result<Option<Bytes>> {
        self.ensure_connection().await?;

        if let Some(subscriber) = &self.subscriber {
            let timeout_duration = timeout.unwrap_or(self.dequeue_timeout);
            let mut batch = subscriber
                .fetch()
                .expires(timeout_duration)
                .max_messages(1)
                .messages()
                .await?;

            if let Some(message) = batch.next().await {
                let message =
                    message.map_err(|e| anyhow::anyhow!("Failed to get message: {}", e))?;
                message
                    .ack()
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to ack message: {}", e))?;
                Ok(Some(message.payload.clone()))
            } else {
                Ok(None)
            }
        } else {
            Err(anyhow::anyhow!("Subscriber not initialized"))
        }
    }

    /// Get the number of messages currently in the queue
    pub async fn get_queue_size(&mut self) -> Result<u64> {
        self.ensure_connection().await?;

        if let Some(client) = &self.client {
            // Get consumer info to get pending messages count
            let stream = client.jetstream().get_stream(&self.stream_name).await?;
            let mut consumer: jetstream::consumer::PullConsumer = stream
                .get_consumer("worker-group")
                .await
                .map_err(|e| anyhow::anyhow!("Failed to get consumer: {}", e))?;
            let info = consumer.info().await?;

            Ok(info.num_pending)
        } else {
            Err(anyhow::anyhow!("Client not connected"))
        }
    }
}

/// Prometheus metrics that mirror the NATS client statistics (in primitive types)
/// to be used for the System Status Server.
///
/// ⚠️  IMPORTANT: These Prometheus Gauges are COPIES of NATS client data, not live references!
///
/// How it works:
/// 1. NATS client provides source data via client.statistics() and connection_state()
/// 2. set_from_client_stats() reads current NATS values and updates these Prometheus Gauges
/// 3. Prometheus scrapes these Gauge values (snapshots, not live data)
///
/// Flow: NATS Client → Client Statistics → set_from_client_stats() → Prometheus Gauge
/// Note: These are snapshots updated when set_from_client_stats() is called.
#[derive(Debug, Clone)]
pub struct DRTNatsPrometheusMetrics {
    nats_client: client::Client,
    /// Number of bytes received (excluding protocol overhead)
    pub in_bytes: IntGauge,
    /// Number of bytes sent (excluding protocol overhead)
    pub out_bytes: IntGauge,
    /// Number of messages received
    pub in_messages: IntGauge,
    /// Number of messages sent
    pub out_messages: IntGauge,
    /// Number of times connection was established
    pub connects: IntGauge,
    /// Current connection state (0 = disconnected, 1 = connected, 2 = reconnecting)
    pub connection_state: IntGauge,
}

impl DRTNatsPrometheusMetrics {
    /// Create a new instance of NATS client metrics using a DistributedRuntime's Prometheus constructors
    pub fn new(drt: &crate::DistributedRuntime, nats_client: client::Client) -> Result<Self> {
        let in_bytes = drt.create_intgauge(
            nats_metrics::IN_TOTAL_BYTES,
            "Total number of bytes received by NATS client",
            &[],
        )?;
        let out_bytes = drt.create_intgauge(
            nats_metrics::OUT_OVERHEAD_BYTES,
            "Total number of bytes sent by NATS client",
            &[],
        )?;
        let in_messages = drt.create_intgauge(
            nats_metrics::IN_MESSAGES,
            "Total number of messages received by NATS client",
            &[],
        )?;
        let out_messages = drt.create_intgauge(
            nats_metrics::OUT_MESSAGES,
            "Total number of messages sent by NATS client",
            &[],
        )?;
        let connects = drt.create_intgauge(
            nats_metrics::CONNECTS,
            "Total number of connections established by NATS client",
            &[],
        )?;
        let connection_state = drt.create_intgauge(
            nats_metrics::CONNECTION_STATE,
            "Current connection state of NATS client (0=disconnected, 1=connected, 2=reconnecting)",
            &[],
        )?;

        Ok(Self {
            nats_client,
            in_bytes,
            out_bytes,
            in_messages,
            out_messages,
            connects,
            connection_state,
        })
    }

    /// Copy statistics from the stored NATS client to these Prometheus metrics
    pub fn set_from_client_stats(&self) {
        let stats = self.nats_client.statistics();

        // Get current values from the client statistics
        let in_bytes = stats.in_bytes.load(Ordering::Relaxed);
        let out_bytes = stats.out_bytes.load(Ordering::Relaxed);
        let in_messages = stats.in_messages.load(Ordering::Relaxed);
        let out_messages = stats.out_messages.load(Ordering::Relaxed);
        let connects = stats.connects.load(Ordering::Relaxed);

        // Get connection state
        let connection_state = match self.nats_client.connection_state() {
            State::Connected => 1,
            // treat Disconnected and Pending as "down"
            State::Disconnected | State::Pending => 0,
        };

        // Update Prometheus metrics
        // Using gauges allows us to set absolute values directly
        self.in_bytes.set(in_bytes as i64);
        self.out_bytes.set(out_bytes as i64);
        self.in_messages.set(in_messages as i64);
        self.out_messages.set(out_messages as i64);
        self.connects.set(connects as i64);
        self.connection_state.set(connection_state);
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use figment::Jail;

    #[test]
    fn test_client_options_builder() {
        Jail::expect_with(|_jail| {
            let opts = ClientOptions::builder().build();
            assert!(opts.is_ok());
            Ok(())
        });

        Jail::expect_with(|jail| {
            jail.set_env("NATS_SERVER", "nats://localhost:5222");
            jail.set_env("NATS_AUTH_USERNAME", "user");
            jail.set_env("NATS_AUTH_PASSWORD", "pass");

            let opts = ClientOptions::builder().build();
            assert!(opts.is_ok());
            let opts = opts.unwrap();

            assert_eq!(opts.server, "nats://localhost:5222");
            assert_eq!(
                opts.auth,
                NatsAuth::UserPass("user".to_string(), "pass".to_string())
            );

            Ok(())
        });

        Jail::expect_with(|jail| {
            jail.set_env("NATS_SERVER", "nats://localhost:5222");
            jail.set_env("NATS_AUTH_USERNAME", "user");
            jail.set_env("NATS_AUTH_PASSWORD", "pass");

            let opts = ClientOptions::builder()
                .server("nats://localhost:6222")
                .auth(NatsAuth::Token("token".to_string()))
                .build();
            assert!(opts.is_ok());
            let opts = opts.unwrap();

            assert_eq!(opts.server, "nats://localhost:6222");
            assert_eq!(opts.auth, NatsAuth::Token("token".to_string()));

            Ok(())
        });
    }
}
