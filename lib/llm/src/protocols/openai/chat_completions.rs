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

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::engines::ValidateRequest;

use super::{
    common_ext::{CommonExt, CommonExtProvider},
    nvext::NvExt,
    nvext::NvExtProvider,
    validate, OpenAIOutputOptionsProvider, OpenAISamplingOptionsProvider,
    OpenAIStopConditionsProvider,
};

mod aggregator;
mod delta;

pub use aggregator::DeltaAggregator;
pub use delta::DeltaGenerator;

/// A request structure for creating a chat completion, extending OpenAI's
/// `CreateChatCompletionRequest` with [`NvExt`] extensions and common fields.
///
/// # Fields
/// - `inner`: The base OpenAI chat completion request, embedded using `serde(flatten)`.
/// - `common`: Common extension fields (ignore_eos, min_tokens) at root level, embedded using `serde(flatten)`.
/// - `nvext`: The optional NVIDIA extension field. See [`NvExt`] for more details.
///   Note: If ignore_eos is specified in both common and nvext, the common (root-level) value takes precedence.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateChatCompletionRequest {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateChatCompletionRequest,

    #[serde(flatten, default)]
    pub common: CommonExt,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

/// A response structure for unary chat completion responses, embedding OpenAI's
/// `CreateChatCompletionResponse`.
///
/// # Fields
/// - `inner`: The base OpenAI unary chat completion response, embedded
///   using `serde(flatten)`.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateChatCompletionResponse {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateChatCompletionResponse,
}

/// A response structure for streamed chat completions, embedding OpenAI's
/// `CreateChatCompletionStreamResponse`.
///
/// # Fields
/// - `inner`: The base OpenAI streaming chat completion response, embedded
///   using `serde(flatten)`.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateChatCompletionStreamResponse {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateChatCompletionStreamResponse,
}

/// Implements `NvExtProvider` for `NvCreateChatCompletionRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateChatCompletionRequest {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    /// Returns `None`, as raw prompt extraction is not implemented.
    fn raw_prompt(&self) -> Option<String> {
        None
    }
}

/// Implements `AnnotationsProvider` for `NvCreateChatCompletionRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the list of annotations from `NvExt`, if present.
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    /// Checks whether a specific annotation exists in the request.
    ///
    /// # Arguments
    /// * `annotation` - A string slice representing the annotation to check.
    ///
    /// # Returns
    /// `true` if the annotation exists, `false` otherwise.
    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

/// Implements `OpenAISamplingOptionsProvider` for `NvCreateChatCompletionRequest`,
/// exposing OpenAI's sampling parameters for chat completion.
impl OpenAISamplingOptionsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the temperature parameter for sampling, if set.
    fn get_temperature(&self) -> Option<f32> {
        self.inner.temperature
    }

    /// Retrieves the top-p (nucleus sampling) parameter, if set.
    fn get_top_p(&self) -> Option<f32> {
        self.inner.top_p
    }

    /// Retrieves the frequency penalty parameter, if set.
    fn get_frequency_penalty(&self) -> Option<f32> {
        self.inner.frequency_penalty
    }

    /// Retrieves the presence penalty parameter, if set.
    fn get_presence_penalty(&self) -> Option<f32> {
        self.inner.presence_penalty
    }

    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Implements `CommonExtProvider` for `NvCreateChatCompletionRequest`,
/// providing access to common extension fields.
impl CommonExtProvider for NvCreateChatCompletionRequest {
    /// Returns a reference to the CommonExt struct.
    fn common_ext(&self) -> Option<&CommonExt> {
        Some(&self.common)
    }

    /// Guided Decoding Options
    fn get_guided_json(&self) -> Option<&serde_json::Value> {
        self.common
            .guided_json
            .as_ref()
            .or_else(|| self.nvext.as_ref().and_then(|nv| nv.guided_json.as_ref()))
    }

    fn get_guided_regex(&self) -> Option<String> {
        self.common
            .guided_regex
            .clone()
            .or_else(|| self.nvext.as_ref().and_then(|nv| nv.guided_regex.clone()))
    }

    fn get_guided_grammar(&self) -> Option<String> {
        self.common
            .guided_grammar
            .clone()
            .or_else(|| self.nvext.as_ref().and_then(|nv| nv.guided_grammar.clone()))
    }

    fn get_guided_choice(&self) -> Option<Vec<String>> {
        self.common
            .guided_choice
            .clone()
            .or_else(|| self.nvext.as_ref().and_then(|nv| nv.guided_choice.clone()))
    }

    fn get_guided_decoding_backend(&self) -> Option<String> {
        self.common.guided_decoding_backend.clone().or_else(|| {
            self.nvext
                .as_ref()
                .and_then(|nv| nv.guided_decoding_backend.clone())
        })
    }
}

/// Implements `OpenAIStopConditionsProvider` for `NvCreateChatCompletionRequest`,
/// providing access to stop conditions that control chat completion behavior.
impl OpenAIStopConditionsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the maximum number of tokens allowed in the response.
    #[allow(deprecated)]
    fn get_max_tokens(&self) -> Option<u32> {
        self.inner.max_completion_tokens.or(self.inner.max_tokens)
    }

    /// Retrieves the minimum number of tokens required in the response.
    /// Returns `min_tokens` Value
    /// `min_tokens` is not an OpenAI-supported parameter.
    fn get_min_tokens(&self) -> Option<u32> {
        self.common.min_tokens
    }

    /// Retrieves the stop conditions that terminate the chat completion response.
    ///
    /// Converts OpenAI's `Stop` enum to a `Vec<String>`, normalizing the representation.
    ///
    /// # Returns
    /// * `Some(Vec<String>)` if stop conditions are set.
    /// * `None` if no stop conditions are defined.
    fn get_stop(&self) -> Option<Vec<String>> {
        self.inner.stop.as_ref().map(|stop| match stop {
            async_openai::types::Stop::String(s) => vec![s.clone()],
            async_openai::types::Stop::StringArray(arr) => arr.clone(),
        })
    }

    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    /// Get ignore_eos from CommonExt.
    fn get_common_ignore_eos(&self) -> Option<bool> {
        self.common.ignore_eos
    }
}

impl OpenAIOutputOptionsProvider for NvCreateChatCompletionRequest {
    fn get_logprobs(&self) -> Option<u32> {
        match self.inner.logprobs {
            Some(true) => match self.inner.top_logprobs {
                Some(top_logprobs) => Some(top_logprobs as u32),
                None => Some(1_u32),
            },
            Some(false) => None,
            None => None,
        }
    }

    fn get_prompt_logprobs(&self) -> Option<u32> {
        None
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        None
    }

    fn get_formatted_prompt(&self) -> Option<bool> {
        None
    }
}

/// Implements `ValidateRequest` for `NvCreateChatCompletionRequest`,
/// allowing us to validate the data.
impl ValidateRequest for NvCreateChatCompletionRequest {
    fn validate(&self) -> Result<(), anyhow::Error> {
        validate::validate_messages(&self.inner.messages)?;
        validate::validate_model(&self.inner.model)?;
        // none for store
        validate::validate_reasoning_effort(&self.inner.reasoning_effort)?;
        validate::validate_metadata(&self.inner.metadata)?;
        validate::validate_frequency_penalty(self.inner.frequency_penalty)?;
        validate::validate_logit_bias(&self.inner.logit_bias)?;
        // none for logprobs
        validate::validate_top_logprobs(self.inner.top_logprobs)?;
        // validate::validate_max_tokens(self.inner.max_tokens)?; // warning depricated field
        validate::validate_max_completion_tokens(self.inner.max_completion_tokens)?;
        validate::validate_n(self.inner.n)?;
        // none for modalities
        // none for prediction
        // none for audio
        validate::validate_presence_penalty(self.inner.presence_penalty)?;
        // none for response_format
        // none for seed
        validate::validate_service_tier(&self.inner.service_tier)?;
        validate::validate_stop(&self.inner.stop)?;
        // none for stream
        // none for stream_options
        validate::validate_temperature(self.inner.temperature)?;
        validate::validate_top_p(self.inner.top_p)?;
        validate::validate_tools(&self.inner.tools.as_deref())?;
        // none for tool_choice
        // none for parallel_tool_calls
        validate::validate_user(self.inner.user.as_deref())?;
        // none for function call
        // none for functions

        Ok(())
    }
}
