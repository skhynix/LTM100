"""Tests for summary/raw report writers."""

from __future__ import annotations

import csv
import json

import pytest

from ltm100.core.op import OpResult, OpType
from ltm100.metrics.aggregate import aggregate
from ltm100.metrics.report import (
    write_raw_ndjson,
    write_summary_csv,
    write_summary_json,
)


def _results():
    return [
        OpResult(OpType.ADD, "u0", 1.0, 1.3, "ok", n_items=5),
        OpResult(OpType.SEARCH, "u1", 2.0, 2.4, "error", error_kind="timeout"),
        OpResult(OpType.SEARCH, "u0", 3.0, 3.2, "ok"),
    ]


def test_write_summary_json_roundtrips(tmp_path):
    summary = aggregate(_results())
    p = tmp_path / "summary.json"
    write_summary_json(summary, p, meta={"users": 2, "seed": 0})
    data = json.loads(p.read_text())
    assert data["meta"]["users"] == 2
    assert data["summary"]["total"] == 3
    assert set(data["summary"]["by_op"]) == {"add", "search"}
    # Overall throughput/QPS at the top level (total / wall_seconds).
    assert data["summary"]["throughput_ops_s"] > 0.0
    assert data["summary"]["qps"] == data["summary"]["throughput_ops_s"]
    # Empty results still expose the overall throughput keys.
    empty = aggregate([])
    assert empty["total"] == 0
    assert empty["throughput_ops_s"] == 0.0
    assert empty["qps"] == 0.0


def test_write_summary_csv_has_rows_per_op(tmp_path):
    summary = aggregate(_results())
    p = tmp_path / "summary.csv"
    write_summary_csv(summary, p)
    with open(p) as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "op_type"
    op_types = {r[0] for r in rows[1:]}
    assert op_types == {"add", "search", "all"}
    # The overall "all" row carries the top-level throughput/qps and blanks the
    # (ambiguous) latency cells.
    all_row = next(r for r in rows[1:] if r[0] == "all")
    assert all_row[1] == "3"  # count
    assert all_row[2] == f"{summary['throughput_ops_s']:.4f}"
    assert all_row[3] == f"{summary['qps']:.4f}"
    assert all_row[4] == ""  # latency_mean blank
    header = rows[0]
    assert all_row[header.index("offered")] == "3"
    assert all_row[header.index("accepted")] == "3"
    assert all_row[header.index("successful")] == "2"
    assert all_row[header.index("rejected")] == "0"
    assert all_row[header.index("errors_by_kind")] == '{"timeout": 1}'


def test_write_raw_ndjson_one_line_per_result(tmp_path):
    p = tmp_path / "raw.ndjson"
    write_raw_ndjson(_results(), p)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["op_type"] == "add"
    assert first["status"] == "ok"
    assert abs(first["latency_ms"] - 300.0) < 1e-6
    err = json.loads(lines[1])
    assert err["status"] == "error"
    assert err["error_kind"] == "timeout"


def test_rejections_do_not_inflate_throughput_or_lower_latency():
    results = [
        OpResult(OpType.SEARCH, "u0", 0.0, 1.0, "ok", n_items=2),
        OpResult(OpType.SEARCH, "u1", 0.2, 0.7, "error", error_kind="timeout"),
        OpResult(
            OpType.SEARCH,
            "u2",
            2.0,
            2.0,
            "rejected",
            error_kind="queue_full",
        ),
    ]
    summary = aggregate(results)
    search = summary["by_op"]["search"]

    assert summary["offered"] == 3
    assert summary["accepted"] == 2
    assert summary["successful"] == 1
    assert summary["errors"] == 1
    assert summary["rejected"] == 1
    assert summary["offered_ops_s"] == 1.5
    assert summary["accepted_ops_s"] == 1.0
    assert summary["successful_ops_s"] == 0.5
    assert summary["rejected_ops_s"] == 0.5
    assert summary["throughput_ops_s"] == summary["successful_ops_s"]
    assert summary["qps"] == summary["successful_ops_s"]
    assert summary["rejection_rate"] == pytest.approx(1 / 3)
    assert summary["error_rate"] == 0.5  # errors / accepted
    assert summary["errors_by_kind"] == {"timeout": 1}

    # Only the successful request contributes to service latency and items.
    assert search["latency_ms"]["mean"] == 1000.0
    assert search["latency_ms"]["p50"] == 1000.0
    assert search["items"]["mean"] == 2.0
    assert search["errors_by_kind"] == {"timeout": 1}
