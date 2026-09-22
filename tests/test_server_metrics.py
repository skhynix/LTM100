"""Tests for the server-side metrics feature.

Covers the Prometheus exposition parser, the before/after diff with reset
detection, bucket-quantile interpolation, the report builder, the snapshot
collector, the runner's measure hooks, and the report writers. No network: the
adapter-side snapshot test uses the same tiny aiohttp server the adapter tests
use.
"""

from __future__ import annotations

import json
import math

import pytest
from aiohttp import web

from ltm100.adapters.backends.memmachine import MemMachineClient
from ltm100.common import MemoryItem
from ltm100.core.config import RunConfig
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import AddLoad
from ltm100.metrics.report import write_server_metrics, write_summary_json
from ltm100.metrics.server_metrics import (
    SnapshotCollector,
    build_report,
    diff,
    finish,
    parse_prometheus,
    quantile,
)

# -- exposition parsing --------------------------------------------------------


_EXPOSITION = """\
# HELP up Target up
# TYPE up gauge
up 1
event_memory_encode_events_phase_seconds_count{phase="embedding"} 10
event_memory_encode_events_phase_seconds_sum{phase="embedding"} 1.25
event_memory_encode_events_phase_seconds_bucket{phase="embedding",le="0.05"} 4
event_memory_encode_events_phase_seconds_bucket{phase="embedding",le="0.1"} 10
event_memory_encode_events_phase_seconds_bucket{phase="embedding",le="+Inf"} 10
http_request_duration_seconds_count{method="POST",path="/api/v2/memories",status="200"} 7
weird_line_with_labels{a="1",b="x, y"} 3
broken series not a number
"""


def test_parse_prometheus_basics():
    samples = parse_prometheus(_EXPOSITION)
    assert samples[("up", ())] == 1.0
    assert (
        samples[
            (
                "event_memory_encode_events_phase_seconds_count",
                (("phase", "embedding"),),
            )
        ]
        == 10.0
    )
    # +Inf normalized so snapshots key identically.
    assert (
        samples[
            (
                "event_memory_encode_events_phase_seconds_bucket",
                (("le", "inf"), ("phase", "embedding")),
            )
        ]
        == 10.0
    )
    # Commas inside quoted label values survive.
    assert samples[("weird_line_with_labels", (("a", "1"), ("b", "x, y")))] == 3.0
    # A line whose value does not parse is skipped, not raised.
    assert not any(name == "broken" for (name, _) in samples)


def test_parse_prometheus_skips_nan_samples():
    # Stale gauges expose NaN; keeping them would poison arithmetic.
    samples = parse_prometheus("g NaN\nh 1")
    assert ("g", ()) not in samples
    assert samples[("h", ())] == 1.0


# -- diff ----------------------------------------------------------------------


def _key(name: str, **labels: str) -> tuple:
    return (name, tuple(sorted(labels.items())))


def test_diff_deltas_and_resets():
    before = {_key("x_count"): 10.0, _key("y_count"): 5.0, _key("z_count"): 3.0}
    after = {
        _key("x_count"): 15.0,  # normal: +5
        _key("y_count"): 2.0,  # negative: reset
        _key("w_count"): 1.0,  # appeared mid-run: unmatched
    }
    deltas, resets = diff(before, after)
    assert deltas[_key("x_count")] == 5.0
    assert _key("y_count") not in deltas and _key("y_count") in resets
    assert _key("w_count") in resets
    assert _key("z_count") in resets  # disappeared (series count shrunk)
    assert _key("x_count") not in resets


# -- quantiles -----------------------------------------------------------------


def test_quantile_linear_interpolation():
    # 10 samples, 5 of them <= 0.05 and all <= 0.1: p50 lands at the 0.05 edge
    # (target 5, cumulative 5 there), p90 interpolates between 0.05 and 0.1.
    buckets = {0.05: 5.0, 0.1: 10.0, math.inf: 10.0}
    p50, beyond = quantile(buckets, 10, 0.5)
    assert not beyond and p50 == pytest.approx(0.05)
    p90, beyond = quantile(buckets, 10, 0.9)
    assert not beyond and p90 == pytest.approx(0.05 + 0.05 * (9 - 5) / (10 - 5))


def test_quantile_zero_count():
    value, beyond = quantile({math.inf: 0.0}, 0, 0.5)
    assert value is None and not beyond


def test_quantile_beyond_buckets():
    buckets = {0.1: 10.0, math.inf: 12.0}
    value, beyond = quantile(buckets, 12, 0.99)
    assert value is None and beyond


# -- report builder -------------------------------------------------------------


def _phase_lines(
    name: str, phase: str, count: int, total: float, edges: dict[str, int]
) -> str:
    out = [
        f'{name}_count{{phase="{phase}"}} {count}',
        f'{name}_sum{{phase="{phase}"}} {total}',
    ]
    cumulative = 0
    for le, n in edges.items():
        cumulative += n
        out.append(f'{name}_bucket{{phase="{phase}",le="{le}"}} {cumulative}')
    out.append(f'{name}_bucket{{phase="{phase}",le="+Inf"}} {count}')
    return "\n".join(out)


def test_build_report_rows_and_notes():
    before = "\n".join(
        [
            _phase_lines(
                "event_memory_encode_events_phase_seconds",
                "embedding",
                2,
                0.2,
                {"0.05": 2, "0.1": 0},
            ),
            # query phases: one not executed during the window (T1 == T0)
            _phase_lines(
                "event_memory_query_phase_seconds",
                "embedding",
                1,
                0.1,
                {"0.05": 1, "0.1": 0},
            ),
            'http_request_duration_seconds_count{method="POST",path="/api/v2/memories",status="200"} 2',
            'http_request_duration_seconds_sum{method="POST",path="/api/v2/memories",status="200"} 0.4',
            'http_request_duration_seconds_count{method="POST",path="/api/v2/memories",status="500"} 1',
            'http_request_duration_seconds_sum{method="POST",path="/api/v2/memories",status="500"} 0.3',
            'http_request_duration_seconds_count{method="POST",path="/memories/search",status="200"} 1',
            'http_request_duration_seconds_sum{method="POST",path="/memories/search",status="200"} 0.5',
        ]
    )
    after = "\n".join(
        [
            _phase_lines(
                "event_memory_encode_events_phase_seconds",
                "embedding",
                6,
                0.6,
                {"0.05": 4, "0.1": 2},
            ),
            _phase_lines(
                "event_memory_query_phase_seconds",
                "embedding",
                1,
                0.1,
                {"0.05": 1, "0.1": 0},
            ),
            # http count grows across both statuses; the row sums over status.
            'http_request_duration_seconds_count{method="POST",path="/api/v2/memories",status="200"} 4',
            'http_request_duration_seconds_sum{method="POST",path="/api/v2/memories",status="200"} 0.8',
            'http_request_duration_seconds_count{method="POST",path="/api/v2/memories",status="500"} 2',
            'http_request_duration_seconds_sum{method="POST",path="/api/v2/memories",status="500"} 0.6',
            # The search path uses the router-relative form here and still matches.
            'http_request_duration_seconds_count{method="POST",path="/memories/search",status="200"} 3',
            'http_request_duration_seconds_sum{method="POST",path="/memories/search",status="200"} 1.3',
        ]
    )
    report = build_report(before, after, window="measured")
    rows = {r["series"]: r for r in report["rows"]}

    enc = rows['encode_phase{phase="embedding"}']
    assert enc["delta_count"] == 4 and enc["mean_s"] == pytest.approx(0.4 / 4)
    assert enc["p50"] == pytest.approx(0.05)

    q = rows['query_phase{phase="embedding"}']
    assert q["delta_count"] is None and q["note"] == "not executed"

    # A report series the scrape never carried is visible as "no data", not 0.
    seg = rows['encode_phase{phase="segmentation"}']
    assert seg["note"] == "no data"

    http_add = rows['http_request{{path suffix "/memories"}}']
    assert http_add["delta_count"] == 3  # 2 + 1 growth summed over status
    assert http_add["delta_sum_s"] == pytest.approx(0.7)
    assert http_add["mean_s"] == pytest.approx(0.7 / 3)
    # The two http rows stay separate populations: add saw 3 requests, search
    # saw its own 2 (the fixture mixes full-path and router-relative forms --
    # both match the anchored suffix).
    http_search = rows['http_request{{path suffix "/memories/search"}}']
    assert http_search["delta_count"] == 2
    assert http_search["delta_sum_s"] == pytest.approx(0.8)

    # The raw block carries every parsed series, not just the report rows.
    assert report["raw"]["before"] and report["raw"]["after"]
    assert report["window"] == "measured" and report["status"] == "ok"


def test_build_report_reset_excluded():
    before = _phase_lines(
        "event_memory_encode_events_phase_seconds", "embedding", 50, 5.0, {"0.1": 50}
    )
    after = _phase_lines(
        "event_memory_encode_events_phase_seconds", "embedding", 3, 0.3, {"0.1": 3}
    )
    report = build_report(before, after, window="measured")
    row = next(
        r for r in report["rows"] if r["series"] == 'encode_phase{phase="embedding"}'
    )
    assert row["delta_count"] is None
    assert row["note"].startswith("counter reset")
    assert any("reset" in w for w in report["warnings"])


def test_finish_failed_capture():
    result = finish(
        {"before": None, "after": "text", "errors": ["before snapshot failed"]},
        window="measured",
    )
    assert result["enabled"] is False
    assert result["status"] == "failed"
    assert result["rows"] == []
    assert any("before snapshot failed" in w for w in result["warnings"])


# -- snapshot collector ----------------------------------------------------------


class _FakeMetricsClient:
    def __init__(self, text=None, error=None):
        self._text, self._error = text, error
        self.opens = 0

    async def __aenter__(self):
        self.opens += 1
        return self

    async def __aexit__(self, *exc):
        return None

    async def server_metrics_snapshot(self):
        if self._error:
            raise self._error
        return self._text


@pytest.mark.asyncio
async def test_collector_captures_both_and_never_raises():
    client = _FakeMetricsClient(text="up 1")
    collector = SnapshotCollector(client)
    await collector.start()
    await collector.end()
    assert collector.before == collector.after == "up 1"
    assert collector.errors == []
    assert client.opens == 2  # each snapshot opens and closes its own session

    bad = SnapshotCollector(_FakeMetricsClient(error=RuntimeError("boom")))
    await bad.start()
    await bad.end()
    assert bad.before is None and bad.after is None
    assert len(bad.errors) == 2


# -- runner measure hooks ---------------------------------------------------------


class _HookBackend:
    name = "fake"
    order: list[str] = []

    async def setup(self, users):
        return None

    async def add(self, user, items):
        _HookBackend.order.append("add")
        return [f"{user}-{i}" for i in range(len(items))]

    async def search(self, user, query):
        return []

    async def teardown(self, users, *, delete):
        return None


class _HookDataset:
    name = "fake"

    def users(self, n_users, *, seed=0):
        return [f"u{i}" for i in range(n_users)]

    def memory_stream(self, user):
        for i in range(5):
            yield MemoryItem(content=f"{user}-{i}", producer=user)


@pytest.mark.asyncio
async def test_measure_hooks_bracket_the_measured_window():
    _HookBackend.order.clear()

    async def start():
        _HookBackend.order.append("start")

    async def end():
        _HookBackend.order.append("end")

    runner = LoadRunner(
        client=_HookBackend(),
        dataset=_HookDataset(),
        scenario=AddLoad(),
        config=RunConfig(users=1, ops=3, seed=0, preingest=True),
        on_measure_start=start,
        on_measure_end=end,
    )
    await runner.run()
    order = _HookBackend.order
    # Hooks fire exactly once each and bracket the measured ops. Pre-ingest
    # (5 memories at batch size 50 = one add call) must land BEFORE the start
    # hook -- the window excludes warmup -- and there are exactly the 3
    # measured adds between the hooks.
    assert order == ["add", "start", "add", "add", "add", "end"]
    assert len(runner.recorder.raw()) == 3


@pytest.mark.asyncio
async def test_failing_hook_does_not_break_the_run():
    async def start():
        raise RuntimeError("metrics endpoint died")

    runner = LoadRunner(
        client=_HookBackend(),
        dataset=_HookDataset(),
        scenario=AddLoad(),
        config=RunConfig(users=1, ops=2, seed=0),
        on_measure_start=start,
    )
    results = await runner.run()
    assert len(results) == 2
    assert all(r.status == "ok" for r in results)


@pytest.mark.asyncio
async def test_no_hooks_is_the_default():
    # Default construction keeps the old behavior exactly.
    runner = LoadRunner(
        client=_HookBackend(),
        dataset=_HookDataset(),
        scenario=AddLoad(),
        config=RunConfig(users=1, ops=1, seed=0),
    )
    assert runner.on_measure_start is None and runner.on_measure_end is None
    assert len(await runner.run()) == 1


# -- adapter snapshot over a fake server -------------------------------------------


def _metrics_app(text: str, fail: bool = False) -> web.Application:
    async def metrics(request: web.Request) -> web.Response:
        if fail:
            return web.Response(status=500, text="nope")
        return web.Response(text=text, content_type="text/plain")

    app = web.Application()
    app.router.add_get("/api/v2/metrics", metrics)
    return app


@pytest.mark.asyncio
async def test_adapter_metrics_snapshot_is_text():
    runner = web.AppRunner(_metrics_app("up 1\n"))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        port = site._server.sockets[0].getsockname()[1]
        async with MemMachineClient(f"http://127.0.0.1:{port}") as client:
            assert client.supports_server_metrics is True
            text = await client.server_metrics_snapshot()
            assert text == "up 1\n"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_adapter_metrics_snapshot_raises_on_error_status():
    runner = web.AppRunner(_metrics_app("", fail=True))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        port = site._server.sockets[0].getsockname()[1]
        async with MemMachineClient(f"http://127.0.0.1:{port}") as client:
            with pytest.raises(RuntimeError):
                await client.server_metrics_snapshot()
    finally:
        await runner.cleanup()


# -- report writers ------------------------------------------------------------------


def _section() -> dict:
    return {
        "enabled": True,
        "window": "measured",
        "status": "ok",
        "warnings": [],
        "rows": [
            {
                "series": 'encode_phase{phase="embedding"}',
                "delta_count": 4,
                "delta_sum_s": 0.4,
                "mean_s": 0.1,
                "p50": 0.05,
                "p90": None,
                "p99": None,
                "note": "p99 beyond buckets",
            }
        ],
        "raw": {"before": [], "after": []},
    }


def test_write_summary_json_inert_without_server_metrics(tmp_path):
    p = tmp_path / "summary.json"
    write_summary_json({"total": 0}, p, meta={"users": 1})
    assert "server_metrics" not in json.loads(p.read_text())


def test_write_summary_json_embeds_section_without_raw(tmp_path):
    p = tmp_path / "summary.json"
    write_summary_json(
        {"total": 0},
        p,
        server_metrics={k: v for k, v in _section().items() if k != "raw"},
    )
    data = json.loads(p.read_text())
    assert data["server_metrics"]["rows"][0]["delta_count"] == 4
    assert "raw" not in data["server_metrics"]


def test_write_server_metrics_files(tmp_path):
    write_server_metrics(_section(), tmp_path)
    csv_text = (tmp_path / "server_metrics.csv").read_text()
    assert csv_text.splitlines()[0] == (
        "series,delta_count,delta_sum_s,mean_s,p50_s,p90_s,p99_s,note"
    )
    # The series name carries quotes, so the CSV writer doubles and wraps them.
    row = csv_text.splitlines()[1]
    assert row.startswith(
        '"encode_phase{phase=""embedding""}",4,0.400000,0.100000,0.050000,,,'
    )
    raw = json.loads((tmp_path / "server_metrics_raw.json").read_text())
    assert raw == {"before": [], "after": []}


# -- CLI plumbing ----------------------------------------------------------------------


def test_cli_parses_server_metrics_flag():
    from ltm100.cli import build_parser

    args = build_parser().parse_args(
        [
            "run",
            "--config",
            "x.yaml",
            "--scenario",
            "add-load",
            "--ops",
            "5",
            "--server-metrics",
        ]
    )
    assert args.server_metrics is True
    args = build_parser().parse_args(
        ["run", "--config", "x.yaml", "--scenario", "add-load", "--ops", "5"]
    )
    assert args.server_metrics is False
