// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod model_manager;
pub use model_manager::{ModelManager, ModelManagerError};

mod model_entry;
pub use model_entry::ModelEntry;

mod watcher;
pub use watcher::{ModelUpdate, ModelWatcher};

/// The root etcd path for ModelEntry
pub const MODEL_ROOT_PATH: &str = "models";
