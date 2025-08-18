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

use crate::config::HealthStatus;
use crate::logging::make_request_span;
use crate::logging::TraceParent;
use crate::metrics::MetricsRegistry;
use crate::traits::DistributedRuntimeProvider;
use axum::{body, http::StatusCode, response::IntoResponse, routing::get, Router};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;
use tokio::{net::TcpListener, task::JoinHandle};
use tokio_util::sync::CancellationToken;
use tower_http::trace::DefaultMakeSpan;
use tower_http::trace::TraceLayer;

/// System status server information containing socket address and handle
#[derive(Debug)]
pub struct SystemStatusServerInfo {
    pub socket_addr: std::net::SocketAddr,
    pub handle: Option<Arc<JoinHandle<()>>>,
}

impl SystemStatusServerInfo {
    pub fn new(socket_addr: std::net::SocketAddr, handle: Option<JoinHandle<()>>) -> Self {
        Self {
            socket_addr,
            handle: handle.map(Arc::new),
        }
    }

    pub fn address(&self) -> String {
        self.socket_addr.to_string()
    }

    pub fn hostname(&self) -> String {
        self.socket_addr.ip().to_string()
    }

    pub fn port(&self) -> u16 {
        self.socket_addr.port()
    }
}

impl Clone for SystemStatusServerInfo {
    fn clone(&self) -> Self {
        Self {
            socket_addr: self.socket_addr,
            handle: self.handle.clone(),
        }
    }
}

/// System status server state containing metrics and uptime tracking
pub struct SystemStatusState {
    // global drt registry is for printing out the entire Prometheus format output
    root_drt: Arc<crate::DistributedRuntime>,
    start_time: OnceLock<Instant>,
    uptime_gauge: prometheus::Gauge,
}

impl SystemStatusState {
    /// Create new system status server state with the provided metrics registry
    pub fn new(drt: Arc<crate::DistributedRuntime>) -> anyhow::Result<Self> {
        // Note: This metric is created at the DRT level (no namespace), so we manually add "dynamo_" prefix
        // to maintain consistency with the project's metric naming convention
        let uptime_gauge = drt.as_ref().create_gauge(
            "dynamo_uptime_seconds",
            "Total uptime of the DistributedRuntime in seconds",
            &[],
        )?;
        let state = Self {
            root_drt: drt,
            start_time: OnceLock::new(),
            uptime_gauge,
        };
        Ok(state)
    }

    /// Initialize the start time (can only be called once)
    pub fn initialize_start_time(&self) -> Result<(), &'static str> {
        self.start_time
            .set(Instant::now())
            .map_err(|_| "Start time already initialized")
    }

    pub fn uptime(&self) -> Result<std::time::Duration, &'static str> {
        self.start_time
            .get()
            .ok_or("Start time not initialized")
            .map(|start_time| start_time.elapsed())
    }

    /// Get a reference to the distributed runtime
    pub fn drt(&self) -> &crate::DistributedRuntime {
        &self.root_drt
    }

    /// Update the uptime gauge with current value
    pub fn update_uptime_gauge(&self) {
        if let Ok(uptime) = self.uptime() {
            let uptime_seconds = uptime.as_secs_f64();
            self.uptime_gauge.set(uptime_seconds);
        } else {
            tracing::warn!("Failed to update uptime gauge: start time not initialized");
        }
    }
}

/// Start system status server with metrics support
pub async fn spawn_system_status_server(
    host: &str,
    port: u16,
    cancel_token: CancellationToken,
    drt: Arc<crate::DistributedRuntime>,
) -> anyhow::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    // Create system status server state with the provided metrics registry
    let server_state = Arc::new(SystemStatusState::new(drt)?);
    let health_path = server_state
        .drt()
        .system_health
        .lock()
        .unwrap()
        .health_path
        .clone();
    let live_path = server_state
        .drt()
        .system_health
        .lock()
        .unwrap()
        .live_path
        .clone();

    // Initialize the start time
    server_state
        .initialize_start_time()
        .map_err(|e| anyhow::anyhow!("Failed to initialize start time: {}", e))?;

    let app = Router::new()
        .route(
            &health_path,
            get({
                let state = Arc::clone(&server_state);
                move || health_handler(state)
            }),
        )
        .route(
            &live_path,
            get({
                let state = Arc::clone(&server_state);
                move || health_handler(state)
            }),
        )
        .route(
            "/metrics",
            get({
                let state = Arc::clone(&server_state);
                move || metrics_handler(state)
            }),
        )
        .fallback(|| async {
            tracing::info!("[fallback handler] called");
            (StatusCode::NOT_FOUND, "Route not found").into_response()
        })
        .layer(TraceLayer::new_for_http().make_span_with(make_request_span));

    let address = format!("{}:{}", host, port);
    tracing::info!("[spawn_system_status_server] binding to: {}", address);

    let listener = match TcpListener::bind(&address).await {
        Ok(listener) => {
            // get the actual address and port, print in debug level
            let actual_address = listener.local_addr()?;
            tracing::info!(
                "[spawn_system_status_server] system status server bound to: {}",
                actual_address
            );
            (listener, actual_address)
        }
        Err(e) => {
            tracing::error!("Failed to bind to address {}: {}", address, e);
            return Err(anyhow::anyhow!("Failed to bind to address: {}", e));
        }
    };
    let (listener, actual_address) = listener;

    let observer = cancel_token.child_token();
    // Spawn the server in the background and return the handle
    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(observer.cancelled_owned())
            .await
        {
            tracing::error!("System status server error: {}", e);
        }
    });

    Ok((actual_address, handle))
}

/// Health handler
#[tracing::instrument(skip_all, level = "trace")]
async fn health_handler(state: Arc<SystemStatusState>) -> impl IntoResponse {
    let (mut healthy, endpoints) = state
        .drt()
        .system_health
        .lock()
        .unwrap()
        .get_health_status();
    let uptime = match state.uptime() {
        Ok(uptime_state) => Some(uptime_state),
        Err(e) => {
            tracing::error!("Failed to get uptime: {}", e);
            healthy = false;
            None
        }
    };

    let healthy_string = if healthy { "ready" } else { "notready" };
    let status_code = if healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let response = json!({
        "status": healthy_string,
        "uptime": uptime,
        "endpoints": endpoints
    });

    tracing::trace!("Response {}", response.to_string());

    (status_code, response.to_string())
}

/// Metrics handler with DistributedRuntime uptime
#[tracing::instrument(skip_all, level = "trace")]
async fn metrics_handler(state: Arc<SystemStatusState>) -> impl IntoResponse {
    // Update the uptime gauge with current value
    state.update_uptime_gauge();

    // Execute all the callbacks starting at the DistributedRuntime level
    assert!(state.drt().basename() == "");
    let callback_results = state
        .drt()
        .execute_metrics_callbacks(&state.drt().hierarchy());
    for result in callback_results {
        if let Err(e) = result {
            tracing::error!("Error executing metrics callback: {}", e);
        }
    }

    // Get all metrics from DistributedRuntime (top-level)
    match state.drt().prometheus_metrics_fmt() {
        Ok(response) => (StatusCode::OK, response),
        Err(e) => {
            tracing::error!("Failed to get metrics from registry: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to get metrics".to_string(),
            )
        }
    }
}

// Regular tests: cargo test system_status_server --lib
// Integration tests: cargo test system_status_server --lib --features integration

#[cfg(test)]
/// Helper function to create a DRT instance for async testing
/// Uses the test-friendly constructor without discovery
async fn create_test_drt_async() -> crate::DistributedRuntime {
    let rt = crate::Runtime::from_current().unwrap();
    crate::DistributedRuntime::from_settings_without_discovery(rt)
        .await
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::tests::load_log;
    use crate::metrics::MetricsRegistry;
    use anyhow::{anyhow, Result};
    use chrono::{DateTime, Utc};
    use jsonschema::{Draft, JSONSchema};
    use rstest::rstest;
    use serde_json::Value;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::sync::Arc;
    use stdio_override::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_http_server_lifecycle() {
        let cancel_token = CancellationToken::new();
        let cancel_token_for_server = cancel_token.clone();

        // Test basic HTTP server lifecycle without DistributedRuntime
        let app = Router::new().route("/test", get(|| async { (StatusCode::OK, "test") }));

        // start HTTP server
        let server_handle = tokio::spawn(async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(cancel_token_for_server.cancelled_owned())
                .await;
        });

        // wait for a while to let the server start
        sleep(Duration::from_millis(100)).await;

        // cancel token
        cancel_token.cancel();

        // wait for the server to shut down
        let result = tokio::time::timeout(Duration::from_secs(5), server_handle).await;
        assert!(
            result.is_ok(),
            "HTTP server should shut down when cancel token is cancelled"
        );
    }

    #[cfg(feature = "integration")]
    #[tokio::test]
    async fn test_runtime_metrics_initialization_and_namespace() {
        // Test that metrics have correct namespace
        let drt = create_test_drt_async().await;
        let runtime_metrics = SystemStatusState::new(Arc::new(drt)).unwrap();

        // Initialize start time
        runtime_metrics.initialize_start_time().unwrap();

        runtime_metrics.uptime_gauge.set(42.0);

        let response = runtime_metrics.drt().prometheus_metrics_fmt().unwrap();
        println!("Full metrics response:\n{}", response);

        // Filter out NATS client metrics for comparison
        use crate::metrics::prometheus_names::nats as nats_metrics;

        let filtered_response: String = response
            .lines()
            .filter(|line| !line.contains(nats_metrics::PREFIX))
            .collect::<Vec<_>>()
            .join("\n");

        let expected = "\
# HELP dynamo_component_dynamo_uptime_seconds Total uptime of the DistributedRuntime in seconds
# TYPE dynamo_component_dynamo_uptime_seconds gauge
dynamo_component_dynamo_uptime_seconds 42";
        assert_eq!(filtered_response, expected);
    }

    #[cfg(feature = "integration")]
    #[tokio::test]
    async fn test_start_time_initialization() {
        // Test that start time can only be initialized once
        let drt = create_test_drt_async().await;
        let runtime_metrics = SystemStatusState::new(Arc::new(drt)).unwrap();

        // First initialization should succeed
        assert!(runtime_metrics.initialize_start_time().is_ok());

        // Second initialization should fail
        assert!(runtime_metrics.initialize_start_time().is_err());

        // Uptime should work after initialization
        let _uptime = runtime_metrics.uptime().unwrap();
        // If we get here, uptime calculation works correctly
    }

    #[rstest]
    #[case("ready", 200, "ready", None, None, 3)]
    #[case("notready", 503, "notready", None, None, 3)]
    #[case("ready", 200, "ready", Some("/custom/health"), Some("/custom/live"), 5)]
    #[case(
        "notready",
        503,
        "notready",
        Some("/custom/health"),
        Some("/custom/live"),
        5
    )]
    #[tokio::test]
    #[cfg(feature = "integration")]
    async fn test_health_endpoints(
        #[case] starting_health_status: &'static str,
        #[case] expected_status: u16,
        #[case] expected_body: &'static str,
        #[case] custom_health_path: Option<&'static str>,
        #[case] custom_live_path: Option<&'static str>,
        #[case] expected_num_tests: usize,
    ) {
        use std::sync::Arc;
        use tokio::time::sleep;
        use tokio_util::sync::CancellationToken;
        // use tokio::io::{AsyncReadExt, AsyncWriteExt};
        // use reqwest for HTTP requests

        // Closure call is needed here to satisfy async_with_vars

        crate::logging::init();

        #[allow(clippy::redundant_closure_call)]
        temp_env::async_with_vars(
            [
                (
                    "DYN_SYSTEM_STARTING_HEALTH_STATUS",
                    Some(starting_health_status),
                ),
                ("DYN_SYSTEM_HEALTH_PATH", custom_health_path),
                ("DYN_SYSTEM_LIVE_PATH", custom_live_path),
            ],
            (async || {
                let runtime = crate::Runtime::from_settings().unwrap();
                let drt = Arc::new(
                    crate::DistributedRuntime::from_settings_without_discovery(runtime)
                        .await
                        .unwrap(),
                );
                let cancel_token = CancellationToken::new();
                let (addr, _) =
                    spawn_system_status_server("127.0.0.1", 0, cancel_token.clone(), drt)
                        .await
                        .unwrap();
                println!("[test] Waiting for server to start...");
                sleep(std::time::Duration::from_millis(1000)).await;
                println!("[test] Server should be up, starting requests...");
                let client = reqwest::Client::new();

                // Prepare test cases
                let mut test_cases = vec![];
                if custom_health_path.is_none() {
                    // When using default paths, test the default paths
                    test_cases.push(("/health", expected_status, expected_body));
                } else {
                    // When using custom paths, default paths should not exist
                    test_cases.push(("/health", 404, "Route not found"));
                    test_cases.push((custom_health_path.unwrap(), expected_status, expected_body));
                }
                if custom_live_path.is_none() {
                    // When using default paths, test the default paths
                    test_cases.push(("/live", expected_status, expected_body));
                } else {
                    // When using custom paths, default paths should not exist
                    test_cases.push(("/live", 404, "Route not found"));
                    test_cases.push((custom_live_path.unwrap(), expected_status, expected_body));
                }
                test_cases.push(("/someRandomPathNotFoundHere", 404, "Route not found"));
                assert_eq!(test_cases.len(), expected_num_tests);

                for (path, expect_status, expect_body) in test_cases {
                    println!("[test] Sending request to {}", path);
                    let url = format!("http://{}{}", addr, path);
                    let response = client.get(&url).send().await.unwrap();
                    let status = response.status();
                    let body = response.text().await.unwrap();
                    println!(
                        "[test] Response for {}: status={}, body={:?}",
                        path, status, body
                    );
                    assert_eq!(
                        status, expect_status,
                        "Response: status={}, body={:?}",
                        status, body
                    );
                    assert!(
                        body.contains(expect_body),
                        "Response: status={}, body={:?}",
                        status,
                        body
                    );
                }
            })(),
        )
        .await;
    }

    #[tokio::test]
    #[cfg(feature = "integration")]
    async fn test_health_endpoint_tracing() -> Result<()> {
        use std::sync::Arc;
        use tokio::time::sleep;
        use tokio_util::sync::CancellationToken;

        // Closure call is needed here to satisfy async_with_vars

        #[allow(clippy::redundant_closure_call)]
        let _ = temp_env::async_with_vars(
            [
                ("DYN_SYSTEM_STARTING_HEALTH_STATUS", Some("ready")),
                ("DYN_LOGGING_JSONL", Some("1")),
                ("DYN_LOG", Some("trace")),
            ],
            (async || {
                // TODO Add proper testing for
                // trace id and parent id

                crate::logging::init();

                let runtime = crate::Runtime::from_settings().unwrap();
                let drt = Arc::new(
                    crate::DistributedRuntime::from_settings_without_discovery(runtime)
                        .await
                        .unwrap(),
                );
                let cancel_token = CancellationToken::new();
                let (addr, _) =
                    spawn_system_status_server("127.0.0.1", 0, cancel_token.clone(), drt)
                        .await
                        .unwrap();
                sleep(std::time::Duration::from_millis(1000)).await;
                let client = reqwest::Client::new();
                for path in [("/health"), ("/live"), ("/someRandomPathNotFoundHere")] {
                    let traceparent_value =
                        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
                    let tracestate_value = "vendor1=opaqueValue1,vendor2=opaqueValue2";
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        reqwest::header::HeaderName::from_static("traceparent"),
                        reqwest::header::HeaderValue::from_str(traceparent_value)?,
                    );
                    headers.insert(
                        reqwest::header::HeaderName::from_static("tracestate"),
                        reqwest::header::HeaderValue::from_str(tracestate_value)?,
                    );
                    let url = format!("http://{}{}", addr, path);
                    let response = client.get(&url).headers(headers).send().await.unwrap();
                    let status = response.status();
                    let body = response.text().await.unwrap();
                    tracing::info!(body = body, status = status.to_string());
                }

                Ok::<(), anyhow::Error>(())
            })(),
        )
        .await;
        Ok(())
    }

    #[cfg(feature = "integration")]
    #[tokio::test]
    async fn test_uptime_without_initialization() {
        // Test that uptime returns an error if start time is not initialized
        let drt = create_test_drt_async().await;
        let runtime_metrics = SystemStatusState::new(Arc::new(drt)).unwrap();

        // This should return an error because start time is not initialized
        let result = runtime_metrics.uptime();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Start time not initialized");
    }

    #[cfg(feature = "integration")]
    #[tokio::test]
    async fn test_spawn_system_status_server_endpoints() {
        // use reqwest for HTTP requests
        temp_env::async_with_vars(
            [("DYN_SYSTEM_STARTING_HEALTH_STATUS", Some("ready"))],
            async {
                let cancel_token = CancellationToken::new();
                let drt = create_test_drt_async().await;
                let (addr, server_handle) =
                    spawn_system_status_server("127.0.0.1", 0, cancel_token.clone(), Arc::new(drt))
                        .await
                        .unwrap();
                println!("[test] Waiting for server to start...");
                sleep(std::time::Duration::from_millis(1000)).await;
                println!("[test] Server should be up, starting requests...");
                let client = reqwest::Client::new();
                for (path, expect_200, expect_body) in [
                    ("/health", true, "ready"),
                    ("/live", true, "ready"),
                    ("/someRandomPathNotFoundHere", false, "Route not found"),
                ] {
                    println!("[test] Sending request to {}", path);
                    let url = format!("http://{}{}", addr, path);
                    let response = client.get(&url).send().await.unwrap();
                    let status = response.status();
                    let body = response.text().await.unwrap();
                    println!(
                        "[test] Response for {}: status={}, body={:?}",
                        path, status, body
                    );
                    if expect_200 {
                        assert_eq!(status, 200, "Response: status={}, body={:?}", status, body);
                    } else {
                        assert_eq!(status, 404, "Response: status={}, body={:?}", status, body);
                    }
                    assert!(
                        body.contains(expect_body),
                        "Response: status={}, body={:?}",
                        status,
                        body
                    );
                }
                cancel_token.cancel();
                match server_handle.await {
                    Ok(_) => println!("[test] Server shut down normally"),
                    Err(e) => {
                        if e.is_panic() {
                            println!("[test] Server panicked: {:?}", e);
                        } else {
                            println!("[test] Server cancelled: {:?}", e);
                        }
                    }
                }
            },
        )
        .await;
    }

    #[cfg(feature = "integration")]
    #[tokio::test]
    async fn test_http_server_basic_functionality() {
        // Test basic HTTP server functionality without requiring etcd
        let cancel_token = CancellationToken::new();
        let cancel_token_for_server = cancel_token.clone();

        // Test basic HTTP server lifecycle
        let app = Router::new().route("/test", get(|| async { (StatusCode::OK, "test") }));

        // start HTTP server
        let server_handle = tokio::spawn(async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(cancel_token_for_server.cancelled_owned())
                .await;
        });

        // wait for a while to let the server start
        sleep(Duration::from_millis(100)).await;

        // cancel token
        cancel_token.cancel();

        // wait for the server to shut down
        let result = tokio::time::timeout(Duration::from_secs(5), server_handle).await;
        assert!(
            result.is_ok(),
            "HTTP server should shut down when cancel token is cancelled"
        );
    }
}
