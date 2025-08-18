# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "CircusController",
    "PlannerConnector",
    "KubernetesConnector",
    "LoadPlannerDefaults",
    "SLAPlannerDefaults",
    "ServiceConfig",
]
# Import the classes
from dynamo.planner.circusd import CircusController
from dynamo.planner.config import ServiceConfig
from dynamo.planner.defaults import LoadPlannerDefaults, SLAPlannerDefaults
from dynamo.planner.kubernetes_connector import KubernetesConnector
from dynamo.planner.planner_connector import PlannerConnector

try:
    from ._version import __version__
except Exception:
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("ai-dynamo")
    except Exception:
        __version__ = "0.0.0+unknown"
