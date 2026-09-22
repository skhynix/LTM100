"""Report writers for benchmark summaries and raw per-request streams.

The summary is the aggregated metrics dict (from aggregate.py) wrapped with run
metadata. Raw per-request results can optionally be streamed to NDJSON for
large-scale runs. JSON is for the summary; CSV mirrors the summary's per-op
rows for spreadsheet use.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ltm100.core.op import OpResult


def write_summary_json(
    summary: dict[str, Any],
    path: str | Path,
    *,
    meta: dict[str, Any] | None = None,
    server_metrics: dict[str, Any] | None = None,
) -> None:
    """Write the aggregated summary (plus optional run metadata) as JSON.

    `server_metrics` is the --server-metrics section (without its `raw` block,
    which belongs in server_metrics_raw.json); None omits the key entirely so
    a flag-free run's summary.json is byte-identical to before the feature.
    """
    payload: dict[str, Any] = {"meta": meta or {}, "summary": summary}
    if server_metrics is not None:
        payload["server_metrics"] = server_metrics
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_server_metrics(server_metrics: dict[str, Any], out_dir: str | Path) -> None:
    """Write the --server-metrics outputs into the run's report directory.

    `server_metrics.csv` mirrors the summary section's rows (the phase/http
    latency table) for spreadsheet use. `server_metrics_raw.json` carries the
    complete parsed snapshots -- every series the server exposed, including
    the secondary component metrics (embedder, vector store, segment store,
    ...) the summary table deliberately leaves out: cause-attribution is a
    reading task for the raw file, not a row per series in the report.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def num(value: Any) -> str:
        if value is None:
            return ""
        return f"{value:.6f}" if isinstance(value, float) else str(value)

    with open(out / "server_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "series",
                "delta_count",
                "delta_sum_s",
                "mean_s",
                "p50_s",
                "p90_s",
                "p99_s",
                "note",
            ]
        )
        for row in server_metrics.get("rows", []):
            w.writerow(
                [
                    row.get("series", ""),
                    num(row.get("delta_count")),
                    num(row.get("delta_sum_s")),
                    num(row.get("mean_s")),
                    num(row.get("p50")),
                    num(row.get("p90")),
                    num(row.get("p99")),
                    row.get("note", ""),
                ]
            )
    raw = server_metrics.get("raw")
    if raw is not None:
        with open(out / "server_metrics_raw.json", "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)


def write_summary_csv(summary: dict[str, Any], path: str | Path) -> None:
    """Write one row per op type with the key metrics, plus an overall row.

    The overall ``all`` row aggregates across op types: its throughput/qps come
    from the top-level summary, and its latency cells are intentionally left
    blank because mixing add/search latencies into one distribution is
    ambiguous (see aggregate.py).
    """
    by_op = summary.get("by_op", {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "op_type",
                "count",
                "throughput_ops_s",
                "qps",
                "latency_mean_ms",
                "latency_p50_ms",
                "latency_p90_ms",
                "latency_p95_ms",
                "latency_p99_ms",
                "latency_max_ms",
                "errors",
                "error_rate",
                "offered",
                "accepted",
                "successful",
                "rejected",
                "offered_ops_s",
                "accepted_ops_s",
                "successful_ops_s",
                "rejected_ops_s",
                "rejection_rate",
                "errors_by_kind",
            ]
        )
        for op_type, m in by_op.items():
            lat = m.get("latency_ms", {})
            w.writerow(
                [
                    op_type,
                    m.get("count", 0),
                    f"{m.get('throughput_ops_s', 0.0):.4f}",
                    f"{m.get('qps', 0.0):.4f}",
                    f"{lat.get('mean', 0.0):.4f}",
                    f"{lat.get('p50', 0.0):.4f}",
                    f"{lat.get('p90', 0.0):.4f}",
                    f"{lat.get('p95', 0.0):.4f}",
                    f"{lat.get('p99', 0.0):.4f}",
                    f"{lat.get('max', 0.0):.4f}",
                    m.get("errors", 0),
                    f"{m.get('error_rate', 0.0):.4f}",
                    m.get("offered", 0),
                    m.get("accepted", 0),
                    m.get("successful", 0),
                    m.get("rejected", 0),
                    f"{m.get('offered_ops_s', 0.0):.4f}",
                    f"{m.get('accepted_ops_s', 0.0):.4f}",
                    f"{m.get('successful_ops_s', 0.0):.4f}",
                    f"{m.get('rejected_ops_s', 0.0):.4f}",
                    f"{m.get('rejection_rate', 0.0):.4f}",
                    json.dumps(m.get("errors_by_kind", {}), sort_keys=True),
                ]
            )
        # Overall row across all op types. Latency cells are blank (ambiguous
        # to mix add/search latencies); throughput/qps come from the top level.
        total = summary.get("total", 0)
        total_errors = sum(m.get("errors", 0) for m in by_op.values())
        w.writerow(
            [
                "all",
                total,
                f"{summary.get('throughput_ops_s', 0.0):.4f}",
                f"{summary.get('qps', 0.0):.4f}",
                "",
                "",
                "",
                "",
                "",
                "",
                total_errors,
                f"{summary.get('error_rate', 0.0):.4f}",
                summary.get("offered", 0),
                summary.get("accepted", 0),
                summary.get("successful", 0),
                summary.get("rejected", 0),
                f"{summary.get('offered_ops_s', 0.0):.4f}",
                f"{summary.get('accepted_ops_s', 0.0):.4f}",
                f"{summary.get('successful_ops_s', 0.0):.4f}",
                f"{summary.get('rejected_ops_s', 0.0):.4f}",
                f"{summary.get('rejection_rate', 0.0):.4f}",
                json.dumps(summary.get("errors_by_kind", {}), sort_keys=True),
            ]
        )


def write_raw_ndjson(results: list[OpResult], path: str | Path) -> None:
    """Stream per-request results as one JSON object per line."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            row = {
                "op_type": r.type.value,
                "user_id": r.user_id,
                "started_at": r.started_at,
                "ended_at": r.ended_at,
                "latency_ms": (r.ended_at - r.started_at) * 1000.0,
                "status": r.status,
                "error_kind": r.error_kind,
                "n_items": r.n_items,
            }
            f.write(json.dumps(row) + "\n")


__all__ = [
    "write_raw_ndjson",
    "write_server_metrics",
    "write_summary_csv",
    "write_summary_json",
]
