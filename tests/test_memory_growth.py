"""Memory-growth sweep orchestration and reporting tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ltm100.adapters.datasets.synthetic import SyntheticAdapter
from ltm100.cli import _run_memory_growth, build_parser
from ltm100.core.memory_growth import (
    NamespacedDataset,
    aggregate_repeats,
    evaluate_repeat,
    parse_memory_counts,
    write_sweep_reports,
)


def _sweep_args(tmp_path: Path, *extra: str):
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset:\n  name: synthetic\n  memories_per_user: 20\n"
        "backend:\n  name: memmachine\n"
    )
    return build_parser().parse_args(
        [
            "sweep",
            "memory-growth",
            "--config",
            str(config),
            "--memory-counts",
            "10,20",
            "--queries-per-user",
            "5",
            "--users",
            "2",
            "--ops",
            "4",
            "--repetitions",
            "2",
            "--output",
            str(tmp_path / "out"),
            *extra,
        ]
    )


def _payload(memory_count: int, *, users: int = 2, empty_rate: float = 0.0):
    return {
        "meta": {
            "preingest_stats": {
                "users": users,
                "input_items": memory_count * users,
                "min_items_per_user": memory_count,
                "max_items_per_user": memory_count,
            }
        },
        "summary": {
            "wall_seconds": 1.5,
            "by_op": {
                "search": {
                    "qps": float(memory_count),
                    "latency_ms": {"p50": 1.0, "p95": 2.0, "p99": 3.0},
                    "error_rate": 0.0,
                    "rejection_rate": 0.0,
                    "items": {"empty_rate": empty_rate},
                }
            },
        },
    }


@pytest.mark.parametrize("seed", [0, 42, -7])
def test_namespaced_dataset_preserves_source_contents_and_changes_identity(seed):
    source = SyntheticAdapter(memories_per_user=3)
    expected_user = source.users(1, seed=seed)[0]
    expected = list(source.memory_stream(expected_user))

    wrapped = NamespacedDataset(source, "point_a_")
    user = wrapped.users(1, seed=seed)[0]
    actual = list(wrapped.memory_stream(user))

    assert user == f"point_a_{expected_user}"
    assert [item.content for item in actual] == [item.content for item in expected]
    assert {item.producer for item in actual} == {user}


@pytest.mark.parametrize("seed", [42, -7])
def test_growth_points_and_repetitions_keep_seeded_query_prefix(seed):
    from itertools import islice

    from ltm100.core.scenarios import SearchLoad

    source = SyntheticAdapter(memories_per_user=20)
    original = source.users(2, seed=seed)
    expected = {u: list(source.memory_stream(u)) for u in original}
    for namespace, count in (
        ("point10_repeat1_", 10),
        ("point20_repeat1_", 20),
        ("point10_repeat2_", 10),
    ):
        wrapped = NamespacedDataset(SyntheticAdapter(memories_per_user=20), namespace)
        users = wrapped.users(2, seed=seed)
        for user, original_user in zip(users, original):
            ingested = list(islice(wrapped.memory_stream(user), count))
            assert [item.content for item in ingested] == [
                item.content for item in expected[original_user][:count]
            ]
            scenario = SearchLoad(query_limit=5)
            operations = list(islice(scenario.plan(user, wrapped, {"seed": seed}), 5))
            assert {op.query.query for op in operations} == {
                item.content for item in expected[original_user][:5]
            }


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "positive integers"),
        ("0,10", "positive integers"),
        ("10,10", "strictly increasing"),
        ("20,10", "strictly increasing"),
        ("10,nope", "integers"),
    ],
)
def test_parse_memory_counts_rejects_invalid_values(value: str, message: str):
    with pytest.raises(ValueError, match=message):
        parse_memory_counts(value)


def test_parse_memory_counts_accepts_increasing_points():
    assert parse_memory_counts("10, 100,1000") == [10, 100, 1000]


def test_evaluate_repeat_checks_accounting_and_search_health():
    metrics, reasons = evaluate_repeat(
        _payload(100), memory_count=100, users=2, max_empty_rate=0.0
    )
    assert reasons == []
    assert metrics["input_items"] == 200
    assert metrics["qps"] == 100.0

    invalid = _payload(100, empty_rate=0.25)
    invalid["meta"]["preingest_stats"]["input_items"] = 199
    invalid["summary"]["by_op"]["search"]["error_rate"] = 0.1
    _, reasons = evaluate_repeat(invalid, memory_count=100, users=2, max_empty_rate=0.2)
    assert any("input_items" in reason for reason in reasons)
    assert any("error_rate" in reason for reason in reasons)
    assert any("empty_rate" in reason for reason in reasons)


def test_aggregate_repeats_uses_only_valid_measurements():
    repeats = [
        {"status": "ok", "metrics": {name: 1.0 for name in _metric_names()}},
        {"status": "invalid", "metrics": {name: 100.0 for name in _metric_names()}},
        {"status": "ok", "metrics": {name: 3.0 for name in _metric_names()}},
    ]

    result = aggregate_repeats(repeats)

    assert result["qps"] == {"median": 2.0, "min": 1.0, "max": 3.0, "range": 2.0}


def _metric_names() -> tuple[str, ...]:
    return (
        "qps",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "empty_rate",
        "error_rate",
        "rejection_rate",
        "wall_seconds",
    )


def test_write_sweep_reports_preserves_statuses(tmp_path):
    manifest = {
        "points": [
            {
                "memory_count": 10,
                "repetitions": [
                    {
                        "repetition": 1,
                        "status": "failed",
                        "metrics": {},
                        "reasons": [],
                        "error": {"kind": "RuntimeError", "message": "boom"},
                        "output": "n_10/repeat_1",
                    }
                ],
            }
        ]
    }

    write_sweep_reports(manifest, tmp_path)

    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    csv_text = (tmp_path / "summary.csv").read_text()
    assert "failed" in csv_text
    assert "RuntimeError: boom" in csv_text


def test_memory_growth_sweep_reuses_run_path_and_aggregates(tmp_path, monkeypatch):
    args = _sweep_args(tmp_path)
    calls = []

    def fake_run(run_args):
        calls.append(run_args)
        output = Path(run_args.output)
        output.mkdir(parents=True)
        payload = _payload(run_args.preingest_items_per_user)
        (output / "summary.json").write_text(json.dumps(payload))
        return 0

    monkeypatch.setattr("ltm100.cli._run", fake_run)

    assert _run_memory_growth(args) == 0

    assert len(calls) == 4
    assert [call.preingest_items_per_user for call in calls] == [10, 10, 20, 20]
    assert {call.query_limit for call in calls} == {5}
    assert {call.scenario for call in calls} == {"search-load"}
    assert {call.model for call in calls} == {"closed"}
    assert len({call._user_namespace for call in calls}) == 4
    manifest = json.loads((Path(args.output) / "manifest.json").read_text())
    assert manifest["status"] == "ok"
    assert manifest["config"]["dataset_config"] == {
        "generator": "legacy-filler-v1",
        "memories_per_user": 20,
        "content_chars": 200,
        "content_chars_scope": "filler",
        "categories": 0,
        "corpus_seed": args.seed,
        "corpus_seed_source": "run_seed",
    }
    assert manifest["points"][0]["aggregate"]["qps"]["median"] == 10.0
    assert manifest["points"][1]["aggregate"]["qps"]["median"] == 20.0


def test_memory_growth_propagates_server_metrics_interval(tmp_path, monkeypatch):
    args = _sweep_args(
        tmp_path,
        "--repetitions",
        "1",
        "--server-metrics",
        "--server-metrics-interval",
        "5",
    )
    calls = []

    def fake_run(run_args):
        calls.append(run_args)
        output = Path(run_args.output)
        output.mkdir(parents=True)
        (output / "summary.json").write_text(
            json.dumps(_payload(run_args.preingest_items_per_user))
        )
        return 0

    monkeypatch.setattr("ltm100.cli._run", fake_run)

    assert _run_memory_growth(args) == 0
    assert {call.server_metrics_interval for call in calls} == {5.0}
    manifest = json.loads((Path(args.output) / "manifest.json").read_text())
    assert manifest["config"]["server_metrics_interval"] == 5.0


def test_memory_growth_interval_requires_server_metrics(tmp_path):
    args = _sweep_args(tmp_path, "--server-metrics-interval", "5")

    with pytest.raises(ValueError, match="requires --server-metrics"):
        _run_memory_growth(args)

    assert not Path(args.output).exists()


def test_memory_growth_sweep_records_failure_and_continues(tmp_path, monkeypatch):
    args = _sweep_args(tmp_path, "--repetitions", "1")
    completed = []

    def fake_run(run_args):
        if run_args.preingest_items_per_user == 10:
            raise RuntimeError("ingest failed")
        completed.append(run_args.preingest_items_per_user)
        output = Path(run_args.output)
        output.mkdir(parents=True)
        (output / "summary.json").write_text(json.dumps(_payload(20)))
        return 0

    monkeypatch.setattr("ltm100.cli._run", fake_run)

    assert _run_memory_growth(args) == 1

    assert completed == [20]
    manifest = json.loads((Path(args.output) / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["points"][0]["repetitions"][0]["error"]["kind"] == "RuntimeError"
    assert manifest["points"][1]["status"] == "ok"


def test_memory_growth_sweep_excludes_invalid_repeat_from_aggregate(
    tmp_path, monkeypatch
):
    args = _sweep_args(tmp_path, "--repetitions", "1")

    def fake_run(run_args):
        output = Path(run_args.output)
        output.mkdir(parents=True)
        payload = _payload(run_args.preingest_items_per_user, empty_rate=0.5)
        (output / "summary.json").write_text(json.dumps(payload))
        return 0

    monkeypatch.setattr("ltm100.cli._run", fake_run)

    assert _run_memory_growth(args) == 1

    manifest = json.loads((Path(args.output) / "manifest.json").read_text())
    assert manifest["status"] == "invalid"
    assert manifest["points"][0]["repetitions"][0]["status"] == "invalid"
    assert manifest["points"][0]["aggregate"] == {}


def test_memory_growth_sweep_rejects_ambiguous_or_unsafe_inputs(tmp_path):
    args = _sweep_args(tmp_path)
    args.queries_per_user = 11
    with pytest.raises(ValueError, match="smallest memory count"):
        _run_memory_growth(args)

    args.queries_per_user = 5
    output = Path(args.output)
    output.mkdir()
    (output / "existing.txt").write_text("keep")
    with pytest.raises(ValueError, match="not empty"):
        _run_memory_growth(args)


def test_memory_growth_sweep_rejects_fixed_shared_project(tmp_path):
    args = _sweep_args(tmp_path)
    Path(args.config).write_text(
        "dataset:\n  name: synthetic\nbackend:\n"
        "  name: memmachine\n  project_id: shared\n"
    )

    with pytest.raises(ValueError, match="cannot isolate corpus-size points"):
        _run_memory_growth(args)


def test_memory_growth_sweep_fails_fast_when_any_user_is_too_short(
    tmp_path, monkeypatch
):
    args = _sweep_args(tmp_path)
    launched = []

    class UnevenDataset:
        name = "uneven"

        def users(self, n_users, *, seed):
            assert n_users == 2
            return ["long", "short"]

        def memory_stream(self, user):
            count = 20 if user == "long" else 19
            for index in range(count):
                yield type("Item", (), {"content": str(index)})()

    monkeypatch.setattr(
        "ltm100.core.memory_growth.build_dataset", lambda cfg: UnevenDataset()
    )
    monkeypatch.setattr("ltm100.cli._run", lambda run_args: launched.append(run_args))

    with pytest.raises(ValueError, match=r"largest memory point 20.*'short'.*19"):
        _run_memory_growth(args)

    assert launched == []
    assert not Path(args.output).exists()


def test_memory_growth_capacity_scan_stops_at_largest_point(tmp_path, monkeypatch):
    args = _sweep_args(tmp_path, "--repetitions", "1")
    yielded = 0

    class BoundedDataset:
        name = "bounded"

        def users(self, n_users, *, seed):
            return [f"u{index}" for index in range(n_users)]

        def memory_stream(self, user):
            nonlocal yielded
            for index in range(20):
                yielded += 1
                yield type("Item", (), {"content": str(index)})()
            raise AssertionError("capacity scan read beyond the largest point")

    def fake_run(run_args):
        output = Path(run_args.output)
        output.mkdir(parents=True)
        payload = _payload(run_args.preingest_items_per_user)
        (output / "summary.json").write_text(json.dumps(payload))
        return 0

    monkeypatch.setattr(
        "ltm100.core.memory_growth.build_dataset", lambda cfg: BoundedDataset()
    )
    monkeypatch.setattr("ltm100.cli._run", fake_run)

    assert _run_memory_growth(args) == 0
    assert yielded == 40
