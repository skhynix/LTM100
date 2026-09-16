"""YAML run configuration and adapter registry resolution.

The YAML holds the stable per-environment bits: which dataset adapter, which
backend adapter, backend endpoint/auth. Per-run parameters (users, scenario,
duration, seed, ...) come from the CLI and are merged in at run time.

Example config:

    dataset:
      name: longmemeval
      split: longmemeval_s_cleaned
      length: 100
    backend:
      name: memmachine
      base_url: http://localhost:8080
      org_prefix: ltm100
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ltm100.adapters.backends.mem0 import Mem0Client
from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.adapters.datasets.longmemeval import LongMemEvalAdapter
from ltm100.adapters.datasets.synthetic import SyntheticAdapter
from ltm100.common import DatasetAdapter, LTMClient


@dataclass
class AdapterConfig:
    name: str
    options: dict[str, Any]


@dataclass
class BenchmarkConfig:
    dataset: AdapterConfig
    backend: AdapterConfig


def load_config(path: str | Path) -> BenchmarkConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _parse_config(raw)


def _parse_config(raw: dict[str, Any]) -> BenchmarkConfig:
    if "dataset" not in raw:
        raise ValueError("config missing 'dataset' section")
    if "backend" not in raw:
        raise ValueError("config missing 'backend' section")
    return BenchmarkConfig(
        dataset=AdapterConfig(
            name=raw["dataset"].get("name"),
            options={k: v for k, v in raw["dataset"].items() if k != "name"},
        ),
        backend=AdapterConfig(
            name=raw["backend"].get("name"),
            options={k: v for k, v in raw["backend"].items() if k != "name"},
        ),
    )


# -- registry ---------------------------------------------------------------

_DATASET_REGISTRY: dict[str, type] = {
    LongMemEvalAdapter.name: LongMemEvalAdapter,
    SyntheticAdapter.name: SyntheticAdapter,
}

_BACKEND_REGISTRY: dict[str, type] = {
    MemMachineClient.name: MemMachineClient,
    Mem0Client.name: Mem0Client,
}


def _mcp_backend() -> type:
    from ltm100.adapters.backends.memmachine_mcp import MemMachineMcpClient

    return MemMachineMcpClient


def build_dataset(cfg: AdapterConfig) -> DatasetAdapter:
    cls = _DATASET_REGISTRY.get(cfg.name)
    if cls is None:
        raise ValueError(f"unknown dataset adapter: {cfg.name}")
    return cls(**cfg.options)


def build_backend(cfg: AdapterConfig) -> LTMClient:
    # Deferred so that fastmcp, which ships only in the [mcp] extra, is
    # required only by a run that actually selects the MCP backend.
    if cfg.name == "memmachine-mcp":
        return _mcp_backend()(**cfg.options)
    cls = _BACKEND_REGISTRY.get(cfg.name)
    if cls is None:
        raise ValueError(f"unknown backend adapter: {cfg.name}")
    return cls(**cfg.options)


__all__ = [
    "AdapterConfig",
    "BenchmarkConfig",
    "build_backend",
    "build_dataset",
    "load_config",
]
