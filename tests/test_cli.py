"""Tests for config parsing, registry resolution, and the CLI parser."""

from __future__ import annotations

import pytest
import yaml

from ltm100.adapters.backends.mem0 import Mem0Client
from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.adapters.datasets.longmemeval import LongMemEvalAdapter
from ltm100.cli import build_parser
from ltm100.config import build_backend, build_dataset, load_config
from ltm100.core.scenarios import get_scenario


def _write_config(tmp_path) -> str:
    cfg = {
        "dataset": {
            "name": "longmemeval",
            "split": "longmemeval_s_cleaned",
            "length": 5,
        },
        "backend": {
            "name": "memmachine",
            "base_url": "http://localhost:8080",
            "org_prefix": "ltm100",
        },
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_load_config_parses_sections(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    assert cfg.dataset.name == "longmemeval"
    assert cfg.dataset.options["split"] == "longmemeval_s_cleaned"
    assert cfg.dataset.options["length"] == 5
    assert cfg.backend.name == "memmachine"
    assert cfg.backend.options["base_url"] == "http://localhost:8080"


def test_build_dataset_resolves_adapter(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    ds = build_dataset(cfg.dataset)
    assert isinstance(ds, LongMemEvalAdapter)
    assert ds.split == "longmemeval_s_cleaned"


def test_build_backend_resolves_adapter(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    backend = build_backend(cfg.backend)
    assert isinstance(backend, MemMachineClient)
    assert backend.org_prefix == "ltm100"


def test_build_backend_resolves_mem0():
    from ltm100.config import AdapterConfig

    backend = build_backend(
        AdapterConfig("mem0", {"base_url": "http://localhost:8888"})
    )
    assert isinstance(backend, Mem0Client)
    assert backend.user_prefix == "ltm100"


def test_load_config_requires_dataset(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"backend": {"name": "memmachine"}}))
    with pytest.raises(ValueError, match="dataset"):
        load_config(str(p))


def test_load_config_requires_backend(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"dataset": {"name": "longmemeval"}}))
    with pytest.raises(ValueError, match="backend"):
        load_config(str(p))


def test_build_dataset_unknown_raises(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    cfg.dataset.name = "nope"
    with pytest.raises(ValueError, match="unknown dataset"):
        build_dataset(cfg.dataset)


def test_cli_run_requires_termination(tmp_path, monkeypatch):
    # The CLI (not argparse) enforces that `run` needs --duration or --ops.
    from ltm100.cli import main

    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "--config",
                _write_config(tmp_path),
                "--scenario",
                "add-load",
            ]
        )


def test_cli_run_parses_args(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--config",
            _write_config(tmp_path),
            "--scenario",
            "chat-replay",
            "--users",
            "50",
            "--duration",
            "60",
            "--seed",
            "7",
            "--global-concurrency",
            "10",
        ]
    )
    assert args.command == "run"
    assert args.scenario == "chat-replay"
    assert args.users == 50
    assert args.duration == 60.0
    assert args.seed == 7
    assert args.global_concurrency == 10


def test_cli_run_parses_open_model_args(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--config",
            _write_config(tmp_path),
            "--scenario",
            "mixed",
            "--users",
            "20",
            "--duration",
            "30",
            "--model",
            "open",
            "--arrival-rate",
            "5.0",
            "--session-ops",
            "10",
            "--queue-bound",
            "4",
            "--search-weight",
            "0.9",
        ]
    )
    assert args.model == "open"
    assert args.arrival_rate == 5.0
    assert args.session_ops == 10
    assert args.queue_bound == 4
    assert args.search_weight == 0.9
    # The open-model flags map onto a valid RunConfig.
    from ltm100.cli import _build_run_config

    run_cfg = _build_run_config(args)
    assert run_cfg.model == "open"
    assert run_cfg.arrival_rate == 5.0
    scenario = get_scenario("mixed", search_weight=args.search_weight)
    assert scenario.search_weight == 0.9  # type: ignore[attr-defined]


def test_cli_scenario_params_forwarded(tmp_path):
    parser = build_parser()
    from ltm100.cli import _build_scenario

    # chat-replay: think + search_every
    args = parser.parse_args(
        [
            "run",
            "--config",
            _write_config(tmp_path),
            "--scenario",
            "chat-replay",
            "--duration",
            "10",
            "--think",
            "0.2",
            "--search-every",
            "5",
        ]
    )
    replay = _build_scenario(args)
    assert replay.think == 0.2  # type: ignore[attr-defined]
    assert replay.search_every == 5  # type: ignore[attr-defined]

    # chat-replay defaults: search_every defaults to 1 (every user turn)
    args = parser.parse_args(
        [
            "run",
            "--config",
            _write_config(tmp_path),
            "--scenario",
            "chat-replay",
            "--duration",
            "10",
        ]
    )
    replay = _build_scenario(args)
    assert replay.search_every == 1  # type: ignore[attr-defined]

    # mixed: search_weight + think
    args = parser.parse_args(
        [
            "run",
            "--config",
            _write_config(tmp_path),
            "--scenario",
            "mixed",
            "--duration",
            "10",
            "--search-weight",
            "0.5",
            "--think",
            "0.1",
        ]
    )
    mixed = _build_scenario(args)
    assert mixed.search_weight == 0.5  # type: ignore[attr-defined]
    assert mixed.think == 0.1  # type: ignore[attr-defined]

    # --top-k forwards to search-issuing scenarios (default 20).
    args = parser.parse_args(
        [
            "run",
            "--config",
            _write_config(tmp_path),
            "--scenario",
            "chat-replay",
            "--duration",
            "10",
            "--top-k",
            "50",
        ]
    )
    assert _build_scenario(args).top_k == 50  # type: ignore[attr-defined]
    args = parser.parse_args(
        [
            "run",
            "--config",
            _write_config(tmp_path),
            "--scenario",
            "search-load",
            "--duration",
            "10",
            "--top-k",
            "8",
        ]
    )
    assert _build_scenario(args).top_k == 8  # type: ignore[attr-defined]
    args = parser.parse_args(
        [
            "run",
            "--config",
            _write_config(tmp_path),
            "--scenario",
            "mixed",
            "--duration",
            "10",
            "--top-k",
            "3",
        ]
    )
    assert _build_scenario(args).top_k == 3  # type: ignore[attr-defined]
    # add-load has no search -> top_k is not forwarded (no attr), but the
    # default flag value is still 20 on the namespace.
    args = parser.parse_args(
        [
            "run",
            "--config",
            _write_config(tmp_path),
            "--scenario",
            "add-load",
            "--duration",
            "10",
        ]
    )
    assert args.top_k == 20
    assert not hasattr(_build_scenario(args), "top_k")


def test_cli_cleanup_subcommand(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        ["cleanup", "--config", _write_config(tmp_path), "--users", "5"]
    )
    assert args.command == "cleanup"
    assert args.users == 5
