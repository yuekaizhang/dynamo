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

// TODO - refactor this entire module
//
// we want to carry forward the concept of live vs ready for the components
// we will want to associate the components cancellation token with the
// component's "service state"

use crate::{
    component::Component,
    error,
    metrics::{prometheus_names, MetricsRegistry},
    traits::*,
    transports::nats,
    utils::stream,
    DistributedRuntime, Result,
};

use async_nats::Message;
use async_stream::try_stream;
use bytes::Bytes;
use derive_getters::Dissolve;
use futures::stream::{StreamExt, TryStreamExt};
use prometheus;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::time::Duration;

pub struct ServiceClient {
    nats_client: nats::Client,
}

impl ServiceClient {
    pub fn new(nats_client: nats::Client) -> Self {
        ServiceClient { nats_client }
    }
}

/// ServiceSet contains a collection of services with their endpoints and metrics
///
/// Tree structure:
/// Structure:
/// - ServiceSet
///   - services: Vec<ServiceInfo>
///     - name: String
///     - id: String
///     - version: String
///     - started: String
///     - endpoints: Vec<EndpointInfo>
///       - name: String
///       - subject: String
///       - data: Option<NatsStatsMetrics>
///         - average_processing_time: f64
///         - last_error: String
///         - num_errors: u64
///         - num_requests: u64
///         - processing_time: u64
///         - queue_group: String
///         - data: serde_json::Value (custom stats)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceSet {
    services: Vec<ServiceInfo>,
}

/// This is a example JSON from `nats req '$SRV.STATS.dynamo_backend'`:
/// {
///   "type": "io.nats.micro.v1.stats_response",
///   "name": "dynamo_backend",
///   "id": "bdu7nA8tbhy9mEkxIWlkBA",
///   "version": "0.0.1",
///   "started": "2025-08-08T05:07:17.720783523Z",
///   "endpoints": [
///     {
///       "name": "dynamo_backend-generate-694d988806b92e39",
///       "subject": "dynamo_backend.generate-694d988806b92e39",
///       "num_requests": 0,
///       "num_errors": 0,
///       "processing_time": 0,
///       "average_processing_time": 0,
///       "last_error": "",
///       "data": {
///         "val": 10
///       },
///       "queue_group": "q"
///     }
///   ]
/// }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    pub name: String,
    pub id: String,
    pub version: String,
    pub started: String,
    pub endpoints: Vec<EndpointInfo>,
}

/// Each endpoint has name, subject, num_requests, num_errors, processing_time, average_processing_time, last_error, queue_group, and data
#[derive(Debug, Clone, Serialize, Deserialize, Dissolve)]
pub struct EndpointInfo {
    pub name: String,
    pub subject: String,

    /// Extra fields that don't fit in EndpointInfo will be flattened into the Metrics struct.
    #[serde(flatten)]
    pub data: Option<NatsStatsMetrics>,
}

impl EndpointInfo {
    pub fn id(&self) -> Result<i64> {
        let id = self
            .subject
            .split('-')
            .next_back()
            .ok_or_else(|| error!("No id found in subject"))?;

        i64::from_str_radix(id, 16).map_err(|e| error!("Invalid id format: {}", e))
    }
}

// TODO: This is _really_ close to the async_nats::service::Stats object,
// but it's missing a few fields like "name", so use a temporary struct
// for easy deserialization. Ideally, this type already exists or can
// be exposed in the library somewhere.
/// Stats structure returned from NATS service API
/// https://github.com/nats-io/nats.rs/blob/main/async-nats/src/service/endpoint.rs
#[derive(Debug, Clone, Serialize, Deserialize, Dissolve)]
pub struct NatsStatsMetrics {
    // Standard NATS Stats Service API fields from $SRV.STATS.<service_name> requests
    pub average_processing_time: u64, // in nanoseconds according to nats-io
    pub last_error: String,
    pub num_errors: u64,
    pub num_requests: u64,
    pub processing_time: u64, // in nanoseconds according to nats-io
    pub queue_group: String,
    // Field containing custom stats handler data
    pub data: serde_json::Value,
}

impl NatsStatsMetrics {
    pub fn decode<T: for<'de> Deserialize<'de>>(self) -> Result<T> {
        serde_json::from_value(self.data).map_err(Into::into)
    }
}

impl ServiceClient {
    pub async fn unary(
        &self,
        subject: impl Into<String>,
        payload: impl Into<Bytes>,
    ) -> Result<Message> {
        let response = self
            .nats_client
            .client()
            .request(subject.into(), payload.into())
            .await?;
        Ok(response)
    }

    pub async fn collect_services(
        &self,
        service_name: &str,
        timeout: Duration,
    ) -> Result<ServiceSet> {
        let sub = self.nats_client.scrape_service(service_name).await?;
        if timeout.is_zero() {
            tracing::warn!("collect_services: timeout is zero");
        }
        if timeout > Duration::from_secs(10) {
            tracing::warn!("collect_services: timeout is greater than 10 seconds");
        }
        let deadline = tokio::time::Instant::now() + timeout;

        let mut services = vec![];
        let mut s = stream::until_deadline(sub, deadline);
        while let Some(message) = s.next().await {
            if message.payload.is_empty() {
                // Expected while we wait for KV metrics in worker to start
                tracing::trace!(service_name, "collect_services: empty payload from nats");
                continue;
            }
            let info = serde_json::from_slice::<ServiceInfo>(&message.payload);
            match info {
                Ok(info) => services.push(info),
                Err(err) => {
                    let payload = String::from_utf8_lossy(&message.payload);
                    tracing::debug!(%err, service_name, %payload, "error decoding service info");
                }
            }
        }

        Ok(ServiceSet { services })
    }
}

impl ServiceSet {
    pub fn into_endpoints(self) -> impl Iterator<Item = EndpointInfo> {
        self.services
            .into_iter()
            .flat_map(|s| s.endpoints.into_iter())
    }

    /// Get a reference to the services in this ServiceSet
    pub fn services(&self) -> &[ServiceInfo] {
        &self.services
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_service_set() {
        let services = vec![
            ServiceInfo {
                name: "service1".to_string(),
                id: "1".to_string(),
                version: "1.0".to_string(),
                started: "2021-01-01".to_string(),
                endpoints: vec![
                    EndpointInfo {
                        name: "endpoint1".to_string(),
                        subject: "subject1".to_string(),
                        data: Some(NatsStatsMetrics {
                            average_processing_time: 100_000, // 0.1ms = 100,000 nanoseconds
                            last_error: "none".to_string(),
                            num_errors: 0,
                            num_requests: 10,
                            processing_time: 100,
                            queue_group: "group1".to_string(),
                            data: serde_json::json!({"key": "value1"}),
                        }),
                    },
                    EndpointInfo {
                        name: "endpoint2-foo".to_string(),
                        subject: "subject2".to_string(),
                        data: Some(NatsStatsMetrics {
                            average_processing_time: 100_000, // 0.1ms = 100,000 nanoseconds
                            last_error: "none".to_string(),
                            num_errors: 0,
                            num_requests: 10,
                            processing_time: 100,
                            queue_group: "group1".to_string(),
                            data: serde_json::json!({"key": "value1"}),
                        }),
                    },
                ],
            },
            ServiceInfo {
                name: "service1".to_string(),
                id: "2".to_string(),
                version: "1.0".to_string(),
                started: "2021-01-01".to_string(),
                endpoints: vec![
                    EndpointInfo {
                        name: "endpoint1".to_string(),
                        subject: "subject1".to_string(),
                        data: Some(NatsStatsMetrics {
                            average_processing_time: 100_000, // 0.1ms = 100,000 nanoseconds
                            last_error: "none".to_string(),
                            num_errors: 0,
                            num_requests: 10,
                            processing_time: 100,
                            queue_group: "group1".to_string(),
                            data: serde_json::json!({"key": "value1"}),
                        }),
                    },
                    EndpointInfo {
                        name: "endpoint2-bar".to_string(),
                        subject: "subject2".to_string(),
                        data: Some(NatsStatsMetrics {
                            average_processing_time: 100_000, // 0.1ms = 100,000 nanoseconds
                            last_error: "none".to_string(),
                            num_errors: 0,
                            num_requests: 10,
                            processing_time: 100,
                            queue_group: "group1".to_string(),
                            data: serde_json::json!({"key": "value2"}),
                        }),
                    },
                ],
            },
        ];

        let service_set = ServiceSet { services };

        let endpoints: Vec<_> = service_set
            .into_endpoints()
            .filter(|e| e.name.starts_with("endpoint2"))
            .collect();

        assert_eq!(endpoints.len(), 2);
    }
}

/// Prometheus metrics for component service statistics (ordered to match NatsStatsMetrics)
///
/// ⚠️  IMPORTANT: These Prometheus Gauges are COPIES of NATS data, not live references!
///
/// How it works:
/// 1. NATS provides source data via NatsStatsMetrics
/// 2. Metrics callbacks read current NATS values and update these Prometheus Gauges
/// 3. Prometheus scrapes these Gauge values (snapshots, not live data)
///
/// Flow: NATS Service → NatsStatsMetrics (Counters) → Metrics Callback → Prometheus Gauge
/// Note: These are snapshots updated when execute_metrics_callbacks() is called.
#[derive(Debug, Clone)]
pub struct ComponentNatsPrometheusMetrics {
    /// Average processing time in milliseconds (maps to: average_processing_time)
    pub avg_processing_ms: prometheus::Gauge,
    /// Total errors across all endpoints (maps to: num_errors)
    pub total_errors: prometheus::IntGauge,
    /// Total requests across all endpoints (maps to: num_requests)
    pub total_requests: prometheus::IntGauge,
    /// Total processing time in milliseconds (maps to: processing_time)
    pub total_processing_ms: prometheus::IntGauge,
    /// Number of active services (derived from ServiceSet.services)
    pub active_services: prometheus::IntGauge,
    /// Number of active endpoints (derived from ServiceInfo.endpoints)
    pub active_endpoints: prometheus::IntGauge,
}

impl ComponentNatsPrometheusMetrics {
    /// Create new ComponentServiceMetrics using Component's DistributedRuntime's Prometheus constructors
    pub fn new(component: &Component) -> Result<Self> {
        let avg_processing_ms = component.create_gauge(
            prometheus_names::nats::AVG_PROCESSING_MS,
            "Average processing time across all component endpoints in milliseconds",
            &[],
        )?;

        let total_errors = component.create_intgauge(
            prometheus_names::nats::TOTAL_ERRORS,
            "Total number of errors across all component endpoints",
            &[],
        )?;

        let total_requests = component.create_intgauge(
            prometheus_names::nats::TOTAL_REQUESTS,
            "Total number of requests across all component endpoints",
            &[],
        )?;

        let total_processing_ms = component.create_intgauge(
            prometheus_names::nats::TOTAL_PROCESSING_MS,
            "Total processing time across all component endpoints in milliseconds",
            &[],
        )?;

        let active_services = component.create_intgauge(
            prometheus_names::nats::ACTIVE_SERVICES,
            "Number of active services in this component",
            &[],
        )?;

        let active_endpoints = component.create_intgauge(
            prometheus_names::nats::ACTIVE_ENDPOINTS,
            "Number of active endpoints across all services",
            &[],
        )?;

        Ok(Self {
            avg_processing_ms,
            total_errors,
            total_requests,
            total_processing_ms,
            active_services,
            active_endpoints,
        })
    }

    /// Update metrics from scraped ServiceSet data
    pub fn update_from_service_set(&self, service_set: &ServiceSet) {
        // Variables ordered to match NatsStatsMetrics fields
        let mut processing_time_samples = 0u64; // for average_processing_time calculation
        let mut total_errors = 0u64; // maps to: num_errors
        let mut total_requests = 0u64; // maps to: num_requests
        let mut total_processing_time_nanos = 0u64; // maps to: processing_time (nanoseconds from NATS)
        let mut endpoint_count = 0u64; // for derived metrics

        let service_count = service_set.services().len() as i64;

        for service in service_set.services() {
            for endpoint in &service.endpoints {
                endpoint_count += 1;

                if let Some(ref stats) = endpoint.data {
                    total_errors += stats.num_errors;
                    total_requests += stats.num_requests;
                    total_processing_time_nanos += stats.processing_time;

                    if stats.num_requests > 0 {
                        processing_time_samples += 1;
                    }
                }
            }
        }

        // Update metrics (ordered to match NatsStatsMetrics fields)
        // Calculate average processing time in milliseconds (maps to: average_processing_time)
        if processing_time_samples > 0 && total_requests > 0 {
            let avg_time_nanos = total_processing_time_nanos as f64 / total_requests as f64;
            let avg_time_ms = avg_time_nanos / 1_000_000.0; // Convert nanoseconds to milliseconds
            self.avg_processing_ms.set(avg_time_ms);
        } else {
            self.avg_processing_ms.set(0.0);
        }

        self.total_errors.set(total_errors as i64); // maps to: num_errors
        self.total_requests.set(total_requests as i64); // maps to: num_requests
        self.total_processing_ms
            .set((total_processing_time_nanos / 1_000_000) as i64); // maps to: processing_time (converted to milliseconds)
        self.active_services.set(service_count); // derived from ServiceSet.services
        self.active_endpoints.set(endpoint_count as i64); // derived from ServiceInfo.endpoints
    }

    /// Reset all metrics to zero. Useful when no data is available or to clear stale values.
    pub fn reset_to_zeros(&self) {
        self.avg_processing_ms.set(0.0);
        self.total_errors.set(0);
        self.total_requests.set(0);
        self.total_processing_ms.set(0);
        self.active_services.set(0);
        self.active_endpoints.set(0);
    }
}
