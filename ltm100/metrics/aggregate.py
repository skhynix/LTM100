"""Post-run aggregation of per-request results into a summary.

Separates offered, accepted, successful, failed, and rejected requests so an
overloaded run cannot make throughput look better or latency look lower merely
by rejecting quickly. The compatibility fields `throughput_ops_s` and `qps`
mean successful throughput. Latency and item statistics use successful requests
only. Per-op latency stays separate because mixing add/search distributions is
ambiguous. Pure function over a list of OpResult.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

from ltm100.core.op import OpResult


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        # Nearest-rank method.
        rank = math.ceil(p / 100 * n)
        rank = max(1, min(rank, n))
        return ordered[rank - 1]

    return {
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": ordered[-1],
    }


def aggregate(results: Iterable[OpResult]) -> dict:
    results = list(results)
    summary = _aggregate(results)
    groups = sorted({result.group for result in results if result.group})
    if groups:
        summary["by_group"] = {
            group: _aggregate([result for result in results if result.group == group])
            for group in groups
        }
    return summary


def _aggregate(results: list[OpResult]) -> dict:
    if not results:
        return {
            "total": 0,
            "throughput_ops_s": 0.0,
            "qps": 0.0,
            "offered": 0,
            "accepted": 0,
            "successful": 0,
            "errors": 0,
            "rejected": 0,
            "offered_ops_s": 0.0,
            "accepted_ops_s": 0.0,
            "successful_ops_s": 0.0,
            "rejected_ops_s": 0.0,
            "rejection_rate": 0.0,
            "errors_by_kind": {},
            "by_op": {},
            "error_rate": 0.0,
            "wall_seconds": 0.0,
        }

    by_op: dict[str, list[OpResult]] = defaultdict(list)
    for r in results:
        by_op[r.type.value].append(r)

    start = min(r.started_at for r in results)
    end = max(r.ended_at for r in results)
    wall = max(end - start, 0.0)

    summary_by_op: dict[str, dict] = {}
    total = 0
    total_errors = 0
    total_rejected = 0
    total_successful = 0
    errors_by_kind: dict[str, int] = defaultdict(int)
    for op_type, items in by_op.items():
        ok = [r for r in items if r.status == "ok"]
        rejected = [r for r in items if r.status == "rejected"]
        errors = [r for r in items if r.status not in ("ok", "rejected")]
        accepted = len(ok) + len(errors)
        latencies_ms = [(r.ended_at - r.started_at) * 1000.0 for r in ok]
        op_errors_by_kind: dict[str, int] = defaultdict(int)
        for r in errors:
            kind = r.error_kind or "unknown"
            op_errors_by_kind[kind] += 1
            errors_by_kind[kind] += 1
        empty = sum(1 for r in ok if r.n_items == 0)
        total += len(items)
        total_successful += len(ok)
        total_errors += len(errors)
        total_rejected += len(rejected)
        summary_by_op[op_type] = {
            "count": len(items),
            "throughput_ops_s": len(ok) / wall if wall > 0 else 0.0,
            "qps": len(ok) / wall if wall > 0 else 0.0,
            "offered": len(items),
            "accepted": accepted,
            "successful": len(ok),
            "errors": len(errors),
            "rejected": len(rejected),
            "offered_ops_s": len(items) / wall if wall > 0 else 0.0,
            "accepted_ops_s": accepted / wall if wall > 0 else 0.0,
            "successful_ops_s": len(ok) / wall if wall > 0 else 0.0,
            "rejected_ops_s": len(rejected) / wall if wall > 0 else 0.0,
            "rejection_rate": len(rejected) / len(items) if items else 0.0,
            "latency_ms": {
                "mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
                **_percentiles(latencies_ms),
            },
            "error_rate": len(errors) / accepted if accepted else 0.0,
            "errors_by_kind": dict(sorted(op_errors_by_kind.items())),
            "items": {
                "mean": sum(r.n_items for r in ok) / len(ok) if ok else 0.0,
                "empty": empty,
                "empty_rate": empty / len(ok) if ok else 0.0,
            },
        }
        _add_error_breakdown(summary_by_op[op_type], errors)

    accepted = total_successful + total_errors
    successful_throughput = total_successful / wall if wall > 0 else 0.0

    summary = {
        "total": total,
        "throughput_ops_s": successful_throughput,
        "qps": successful_throughput,
        "offered": total,
        "accepted": accepted,
        "successful": total_successful,
        "errors": total_errors,
        "rejected": total_rejected,
        "offered_ops_s": total / wall if wall > 0 else 0.0,
        "accepted_ops_s": accepted / wall if wall > 0 else 0.0,
        "successful_ops_s": successful_throughput,
        "rejected_ops_s": total_rejected / wall if wall > 0 else 0.0,
        "rejection_rate": total_rejected / total if total else 0.0,
        "errors_by_kind": dict(sorted(errors_by_kind.items())),
        "by_op": summary_by_op,
        "error_rate": total_errors / accepted if accepted else 0.0,
        "wall_seconds": round(wall, 6),
    }
    _add_error_breakdown(
        summary, [r for r in results if r.status not in ("ok", "rejected")]
    )
    return summary


def _add_error_breakdown(summary: dict, errors: list[OpResult]) -> None:
    # Legacy callers can still supply results without diagnostics. Do not
    # change their summaries. Once details exist, include undetailed failures
    # as "other" so breakdown counts reconcile with the existing error count.
    if not any(r.error_detail is not None for r in errors):
        return
    counts: dict[tuple, dict] = {}
    for result in errors:
        detail = result.error_detail
        row = {
            "op_type": result.type.value,
            "error_kind": result.error_kind or "unknown",
            "category": detail.category if detail else "other",
        }
        if detail is not None:
            row.update(
                (key, value)
                for key, value in detail.as_dict().items()
                if key not in {"message", "timeout_seconds", "category"}
            )
        key = tuple(row.items())
        if key not in counts:
            counts[key] = {**row, "count": 0}
        counts[key]["count"] += 1
    summary["error_breakdown"] = [counts[key] for key in sorted(counts, key=repr)]


__all__ = ["aggregate"]
