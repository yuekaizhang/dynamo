// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::get, Router};
use prometheus::{Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

pub use prometheus::Registry;

use super::RouteDoc;

// Default metric prefix
pub const FRONTEND_METRIC_PREFIX: &str = "dynamo_frontend";

// Environment variable that overrides the default metric prefix if provided
pub const METRICS_PREFIX_ENV: &str = "DYN_METRICS_PREFIX";

/// Value for the `status` label in the request counter for successful requests
pub const REQUEST_STATUS_SUCCESS: &str = "success";

/// Value for the `status` label in the request counter if the request failed
pub const REQUEST_STATUS_ERROR: &str = "error";

/// Partial value for the `type` label in the request counter for streaming requests
pub const REQUEST_TYPE_STREAM: &str = "stream";

/// Partial value for the `type` label in the request counter for unary requests
pub const REQUEST_TYPE_UNARY: &str = "unary";

fn sanitize_prometheus_prefix(raw: &str) -> String {
    // Prometheus metric name pattern: [a-zA-Z_:][a-zA-Z0-9_:]*
    let mut s: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == ':' {
                c
            } else {
                '_'
            }
        })
        .collect();

    if s.is_empty() {
        return FRONTEND_METRIC_PREFIX.to_string();
    }

    let first = s.as_bytes()[0];
    let valid_first = first.is_ascii_alphabetic() || first == b'_' || first == b':';
    if !valid_first {
        s.insert(0, '_');
    }
    s
}

pub struct Metrics {
    request_counter: IntCounterVec,
    inflight_gauge: IntGaugeVec,
    request_duration: HistogramVec,
    input_sequence_length: HistogramVec,
    output_sequence_length: HistogramVec,
    time_to_first_token: HistogramVec,
    inter_token_latency: HistogramVec,
}

/// RAII object for inflight gauge and request counters
/// If this object is dropped without calling `mark_ok`, then the request will increment
/// the request counter with the `status` label with [`REQUEST_STATUS_ERROR`]; otherwise, it will increment
/// the counter with `status` label [`REQUEST_STATUS_SUCCESS`]
pub struct InflightGuard {
    metrics: Arc<Metrics>,
    model: String,
    endpoint: Endpoint,
    request_type: RequestType,
    status: Status,
    timer: Instant,
}

/// Requests will be logged by the type of endpoint hit
/// This will include llamastack in the future
pub enum Endpoint {
    /// OAI Completions
    Completions,

    /// OAI Chat Completions
    ChatCompletions,

    /// OAI Embeddings
    Embeddings,

    /// OAI Responses
    Responses,
}

/// Metrics for the HTTP service
pub enum RequestType {
    /// SingleIn / SingleOut
    Unary,

    /// SingleIn / ManyOut
    Stream,
}

/// Status
pub enum Status {
    Success,
    Error,
}

/// Track response-specific metrics
pub struct ResponseMetricCollector {
    metrics: Arc<Metrics>,
    model: String,
    start_time: Instant,
    // we use is_first_token to distinguish TTFT from ITL. It is true by default and
    // flipped to false when the first token is returned and TTFT is published.
    is_first_token: bool,
    // we track the last response time so that ITL for the newly returned tokens can
    // be computed.
    last_response_time: Option<Duration>,
    osl: usize,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// Create Metrics with the standard prefix defined by [`FRONTEND_METRIC_PREFIX`] or specify custom prefix via the following environment variable:
    /// - `DYN_METRICS_PREFIX`: Override the default metrics prefix
    ///
    /// The following metrics will be created with the configured prefix:
    /// - `{prefix}_requests_total` - IntCounterVec for the total number of requests processed
    /// - `{prefix}_inflight_requests` - IntGaugeVec for the number of inflight requests
    /// - `{prefix}_request_duration_seconds` - HistogramVec for the duration of requests
    /// - `{prefix}_input_sequence_tokens` - HistogramVec for input sequence length in tokens
    /// - `{prefix}_output_sequence_tokens` - HistogramVec for output sequence length in tokens
    /// - `{prefix}_time_to_first_token_seconds` - HistogramVec for time to first token in seconds
    /// - `{prefix}_inter_token_latency_seconds` - HistogramVec for inter-token latency in seconds
    pub fn new() -> Self {
        let raw_prefix = std::env::var(METRICS_PREFIX_ENV)
            .unwrap_or_else(|_| FRONTEND_METRIC_PREFIX.to_string());
        let prefix = sanitize_prometheus_prefix(&raw_prefix);
        if prefix != raw_prefix {
            tracing::warn!(
                raw=%raw_prefix,
                sanitized=%prefix,
                env=%METRICS_PREFIX_ENV,
                "Sanitized HTTP metrics prefix"
            );
        }
        let frontend_metric_name = |suffix: &str| format!("{}_{}", &prefix, suffix);

        let request_counter = IntCounterVec::new(
            Opts::new(
                frontend_metric_name("requests_total"),
                "Total number of LLM requests processed",
            ),
            &["model", "endpoint", "request_type", "status"],
        )
        .unwrap();

        let inflight_gauge = IntGaugeVec::new(
            Opts::new(
                frontend_metric_name("inflight_requests"),
                "Number of inflight requests",
            ),
            &["model"],
        )
        .unwrap();

        let buckets = vec![0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];

        let request_duration = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name("request_duration_seconds"),
                "Duration of LLM requests",
            )
            .buckets(buckets),
            &["model"],
        )
        .unwrap();

        let input_sequence_length = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name("input_sequence_tokens"),
                "Input sequence length in tokens",
            )
            .buckets(vec![
                0.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0, 64000.0,
                128000.0,
            ]),
            &["model"],
        )
        .unwrap();

        let output_sequence_length = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name("output_sequence_tokens"),
                "Output sequence length in tokens",
            )
            .buckets(vec![
                0.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0,
            ]),
            &["model"],
        )
        .unwrap();

        let time_to_first_token = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name("time_to_first_token_seconds"),
                "Time to first token in seconds",
            )
            .buckets(vec![
                0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0,
                60.0, 120.0, 240.0, 480.0,
            ]),
            &["model"],
        )
        .unwrap();

        let inter_token_latency = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name("inter_token_latency_seconds"),
                "Inter-token latency in seconds",
            )
            .buckets(vec![
                0.0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0,
            ]),
            &["model"],
        )
        .unwrap();

        Metrics {
            request_counter,
            inflight_gauge,
            request_duration,
            input_sequence_length,
            output_sequence_length,
            time_to_first_token,
            inter_token_latency,
        }
    }

    /// Get the number of successful requests for the given dimensions:
    /// - model
    /// - endpoint (completions/chat_completions)
    /// - request type (unary/stream)
    /// - status (success/error)
    pub fn get_request_counter(
        &self,
        model: &str,
        endpoint: &Endpoint,
        request_type: &RequestType,
        status: &Status,
    ) -> u64 {
        self.request_counter
            .with_label_values(&[
                model,
                endpoint.as_str(),
                request_type.as_str(),
                status.as_str(),
            ])
            .get()
    }

    /// Increment the counter for requests for the given dimensions:
    /// - model
    /// - endpoint (completions/chat_completions)
    /// - request type (unary/stream)
    /// - status (success/error)
    fn inc_request_counter(
        &self,
        model: &str,
        endpoint: &Endpoint,
        request_type: &RequestType,
        status: &Status,
    ) {
        self.request_counter
            .with_label_values(&[
                model,
                endpoint.as_str(),
                request_type.as_str(),
                status.as_str(),
            ])
            .inc()
    }

    /// Get the number if inflight requests for the given model
    pub fn get_inflight_count(&self, model: &str) -> i64 {
        self.inflight_gauge.with_label_values(&[model]).get()
    }

    fn inc_inflight_gauge(&self, model: &str) {
        self.inflight_gauge.with_label_values(&[model]).inc()
    }

    fn dec_inflight_gauge(&self, model: &str) {
        self.inflight_gauge.with_label_values(&[model]).dec()
    }

    pub fn register(&self, registry: &Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.request_counter.clone()))?;
        registry.register(Box::new(self.inflight_gauge.clone()))?;
        registry.register(Box::new(self.request_duration.clone()))?;
        registry.register(Box::new(self.input_sequence_length.clone()))?;
        registry.register(Box::new(self.output_sequence_length.clone()))?;
        registry.register(Box::new(self.time_to_first_token.clone()))?;
        registry.register(Box::new(self.inter_token_latency.clone()))?;
        Ok(())
    }

    /// Create a new [`InflightGuard`] for the given model and annotate if its a streaming request,
    /// and the kind of endpoint that was hit
    ///
    /// The [`InflightGuard`] is an RAII object will handle incrementing the inflight gauge and
    /// request counters.
    pub fn create_inflight_guard(
        self: Arc<Self>,
        model: &str,
        endpoint: Endpoint,
        streaming: bool,
    ) -> InflightGuard {
        let request_type = if streaming {
            RequestType::Stream
        } else {
            RequestType::Unary
        };

        InflightGuard::new(
            self.clone(),
            model.to_string().to_lowercase(),
            endpoint,
            request_type,
        )
    }

    /// Create a new [`ResponseMetricCollector`] for collecting per-response metrics (i.e., TTFT, ITL)
    pub fn create_response_collector(self: Arc<Self>, model: &str) -> ResponseMetricCollector {
        ResponseMetricCollector::new(self, model.to_string().to_lowercase())
    }
}

impl InflightGuard {
    fn new(
        metrics: Arc<Metrics>,
        model: String,
        endpoint: Endpoint,
        request_type: RequestType,
    ) -> Self {
        // Start the timer
        let timer = Instant::now();

        // Increment the inflight gauge when the guard is created
        metrics.inc_inflight_gauge(&model);

        // Return the RAII Guard
        InflightGuard {
            metrics,
            model,
            endpoint,
            request_type,
            status: Status::Error,
            timer,
        }
    }

    pub(crate) fn mark_ok(&mut self) {
        self.status = Status::Success;
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        // Decrement the gauge when the guard is dropped
        self.metrics.dec_inflight_gauge(&self.model);

        // the frequency on incrementing the full request counter is relatively low
        // if we were incrementing the counter on every forward pass, we'd use static CounterVec or
        // discrete counter object without the more costly lookup required for the following calls
        self.metrics.inc_request_counter(
            &self.model,
            &self.endpoint,
            &self.request_type,
            &self.status,
        );

        // Record the duration of the request
        self.metrics
            .request_duration
            .with_label_values(&[&self.model])
            .observe(self.timer.elapsed().as_secs_f64());
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Endpoint::Completions => write!(f, "completions"),
            Endpoint::ChatCompletions => write!(f, "chat_completions"),
            Endpoint::Embeddings => write!(f, "embeddings"),
            Endpoint::Responses => write!(f, "responses"),
        }
    }
}

impl Endpoint {
    pub fn as_str(&self) -> &'static str {
        match self {
            Endpoint::Completions => "completions",
            Endpoint::ChatCompletions => "chat_completions",
            Endpoint::Embeddings => "embeddings",
            Endpoint::Responses => "responses",
        }
    }
}

impl RequestType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RequestType::Unary => REQUEST_TYPE_UNARY,
            RequestType::Stream => REQUEST_TYPE_STREAM,
        }
    }
}

impl Status {
    pub fn as_str(&self) -> &'static str {
        match self {
            Status::Success => REQUEST_STATUS_SUCCESS,
            Status::Error => REQUEST_STATUS_ERROR,
        }
    }
}

impl ResponseMetricCollector {
    fn new(metrics: Arc<Metrics>, model: String) -> Self {
        ResponseMetricCollector {
            metrics,
            model,
            is_first_token: true,
            last_response_time: None,
            start_time: Instant::now(),
            osl: 0,
        }
    }

    /// Observe the current output sequence length
    pub fn observe_current_osl(&mut self, osl: usize) {
        self.osl = osl;
    }

    /// Observe a response with input sequence length and number of new tokens
    pub fn observe_response(&mut self, isl: usize, num_tokens: usize) {
        if num_tokens == 0 {
            return;
        }

        if self.is_first_token {
            // NOTE: when there are multiple tokens in the first response,
            // we use the full response time as TTFT and ignore the ITL
            self.is_first_token = false;

            // Publish TTFT
            let ttft = self.start_time.elapsed().as_secs_f64();
            self.metrics
                .time_to_first_token
                .with_label_values(&[&self.model])
                .observe(ttft);

            // Publish ISL
            // TODO: publish ISL as soon as the tokenization process completes
            self.metrics
                .input_sequence_length
                .with_label_values(&[&self.model])
                .observe(isl as f64);
        }

        let current_duration = self.start_time.elapsed();

        if let Some(last_response_time) = self.last_response_time {
            let response_duration = current_duration - last_response_time;
            let itl = response_duration.as_secs_f64() / num_tokens as f64;
            for _ in 0..num_tokens {
                self.metrics
                    .inter_token_latency
                    .with_label_values(&[&self.model])
                    .observe(itl);
            }
        }

        self.last_response_time = Some(current_duration);
    }
}

impl Drop for ResponseMetricCollector {
    fn drop(&mut self) {
        // Publish final OSL when the collector is dropped
        self.metrics
            .output_sequence_length
            .with_label_values(&[&self.model])
            .observe(self.osl as f64);
    }
}

/// Create a new router with the given path
pub fn router(registry: Registry, path: Option<String>) -> (Vec<RouteDoc>, Router) {
    let registry = Arc::new(registry);
    let path = path.unwrap_or_else(|| "/metrics".to_string());
    let doc = RouteDoc::new(axum::http::Method::GET, &path);
    let route = Router::new()
        .route(&path, get(handler_metrics))
        .with_state(registry);
    (vec![doc], route)
}

/// Metrics Handler
async fn handler_metrics(State(registry): State<Arc<Registry>>) -> impl IntoResponse {
    let encoder = prometheus::TextEncoder::new();
    let metric_families = registry.gather();
    let mut buffer = vec![];
    if encoder.encode(&metric_families, &mut buffer).is_err() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to encode metrics",
        )
            .into_response();
    }

    let metrics = match String::from_utf8(buffer) {
        Ok(metrics) => metrics,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to encode metrics",
            )
                .into_response()
        }
    };

    (StatusCode::OK, metrics).into_response()
}
