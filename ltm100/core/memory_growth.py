"""Memory-growth sweep helpers.

The sweep itself is an orchestration layer over the ordinary ``search-load``
run.  This module keeps the experiment-specific concerns out of the runner:
point isolation, validity checks, cross-repetition aggregation, and the root
manifest/CSV report.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import datetime, timezone
from itertools import islice, pairwise
from pathlib import Path
from statistics import median
from typing import Any
from uuid import uuid4

from ltm100.common import DatasetAdapter, MemoryItem, UserId
from ltm100.config import BenchmarkConfig, build_dataset, dataset_settings, load_config

logger = logging.getLogger(__name__)

_METRICS = (
    "qps",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "empty_rate",
    "error_rate",
    "rejection_rate",
    "wall_seconds",
)


class NamespacedDataset:
    """Give a run fresh backend identities without changing its source data.

    The wrapped adapter still receives its original user ids, so every sweep
    point sees byte-identical corpus and query contents.  Only the user and
    producer identities presented to the backend are prefixed.
    """

    def __init__(self, dataset: DatasetAdapter, namespace: str) -> None:
        self._dataset = dataset
        self._namespace = namespace
        self._users: dict[UserId, UserId] = {}
        self.name = dataset.name

    def users(self, n_users: int, *, seed: int) -> list[UserId]:
        source = self._dataset.users(n_users, seed=seed)
        self._users = {f"{self._namespace}{user}": user for user in source}
        return list(self._users)

    def _source_user(self, user: UserId) -> UserId:
        try:
            return self._users[user]
        except KeyError as error:
            raise ValueError(f"unknown namespaced user: {user!r}") from error

    @staticmethod
    def _item(item: MemoryItem, user: UserId) -> MemoryItem:
        return replace(item, producer=user)

    def memory_stream(self, user: UserId) -> Iterator[MemoryItem]:
        source = self._source_user(user)
        for item in self._dataset.memory_stream(source):
            yield self._item(item, user)


def parse_memory_counts(value: str) -> list[int]:
    """Parse a strictly increasing comma-separated list of positive counts."""
    try:
        counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise ValueError("--memory-counts must contain integers") from error
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("--memory-counts must contain positive integers")
    if any(right <= left for left, right in pairwise(counts)):
        raise ValueError("--memory-counts must be strictly increasing")
    return counts


def evaluate_repeat(
    payload: dict[str, Any],
    *,
    memory_count: int,
    users: int,
    max_empty_rate: float,
) -> tuple[dict[str, float | int], list[str]]:
    """Extract comparable search metrics and label an invalid measurement."""
    meta = payload.get("meta", {})
    summary = payload.get("summary", {})
    search = summary.get("by_op", {}).get("search")
    if not isinstance(search, dict):
        return {}, ["measured run produced no search results"]

    preingest = meta.get("preingest_stats") or {}
    expected_items = memory_count * users
    reasons: list[str] = []
    if preingest.get("users") != users:
        reasons.append(
            f"pre-ingest users {preingest.get('users')!r} != expected {users}"
        )
    if preingest.get("input_items") != expected_items:
        reasons.append(
            "pre-ingest input_items "
            f"{preingest.get('input_items')!r} != expected {expected_items}"
        )
    for field in ("min_items_per_user", "max_items_per_user"):
        if preingest.get(field) != memory_count:
            reasons.append(
                f"pre-ingest {field} {preingest.get(field)!r} != "
                f"expected {memory_count}"
            )

    items = search.get("items", {})
    latency = search.get("latency_ms", {})
    metrics: dict[str, float | int] = {
        "input_items": int(preingest.get("input_items") or 0),
        "qps": float(search.get("qps", 0.0)),
        "latency_p50_ms": float(latency.get("p50", 0.0)),
        "latency_p95_ms": float(latency.get("p95", 0.0)),
        "latency_p99_ms": float(latency.get("p99", 0.0)),
        "empty_rate": float(items.get("empty_rate", 0.0)),
        "error_rate": float(search.get("error_rate", 0.0)),
        "rejection_rate": float(search.get("rejection_rate", 0.0)),
        "wall_seconds": float(summary.get("wall_seconds", 0.0)),
    }
    if metrics["error_rate"] > 0:
        reasons.append(f"search error_rate is {metrics['error_rate']:.6f}")
    if metrics["rejection_rate"] > 0:
        reasons.append(f"search rejection_rate is {metrics['rejection_rate']:.6f}")
    if metrics["empty_rate"] > max_empty_rate:
        reasons.append(
            f"search empty_rate {metrics['empty_rate']:.6f} exceeds "
            f"{max_empty_rate:.6f}"
        )
    return metrics, reasons


def aggregate_repeats(repeats: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate valid repetitions without hiding invalid or failed ones."""
    valid = [repeat for repeat in repeats if repeat.get("status") == "ok"]
    result: dict[str, dict[str, float]] = {}
    for name in _METRICS:
        values = [float(repeat["metrics"][name]) for repeat in valid]
        if values:
            low = min(values)
            high = max(values)
            result[name] = {
                "median": median(values),
                "min": low,
                "max": high,
                "range": high - low,
            }
    return result


def write_sweep_reports(manifest: dict[str, Any], output: str | Path) -> None:
    """Atomically refresh the root JSON manifest and repetition CSV."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    temporary = out / "manifest.json.tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    temporary.replace(manifest_path)

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "memory_count",
                "repetition",
                "status",
                "input_items",
                *_METRICS,
                "reasons",
                "error",
                "output",
            ]
        )
        for point in manifest.get("points", []):
            for repeat in point.get("repetitions", []):
                metrics = repeat.get("metrics", {})
                error = repeat.get("error", {})
                writer.writerow(
                    [
                        point["memory_count"],
                        repeat["repetition"],
                        repeat["status"],
                        metrics.get("input_items", ""),
                        *[metrics.get(name, "") for name in _METRICS],
                        "; ".join(repeat.get("reasons", [])),
                        (
                            f"{error.get('kind')}: {error.get('message')}"
                            if error
                            else ""
                        ),
                        repeat["output"],
                    ]
                )


def _run_args(
    args: argparse.Namespace,
    *,
    memory_count: int,
    repetition: int,
    namespace: str,
    output: Path,
) -> argparse.Namespace:
    """Translate one sweep point into an ordinary ``search-load`` run."""
    return argparse.Namespace(
        config=args.config,
        users=args.users,
        seed=args.seed,
        scenario="search-load",
        duration=args.duration,
        ops=args.ops,
        global_concurrency=args.global_concurrency,
        warmup=args.warmup,
        preingest=True,
        preingest_fraction=1.0,
        preingest_items_per_user=memory_count,
        rampup=args.rampup,
        model="closed",
        arrival_rate=0.0,
        session_ops=0,
        queue_bound=0,
        search_weight=0.8,
        top_k=args.top_k,
        query_limit=args.queries_per_user,
        think=0.05,
        search_every=1,
        chat_profile=None,
        answer_time=0.0,
        user_gap=0.0,
        answer_time_variation=None,
        user_gap_variation=None,
        expand=args.expand,
        filter=args.filter,
        procs=args.procs,
        server_metrics=args.server_metrics,
        server_metrics_interval=getattr(args, "server_metrics_interval", None),
        output=str(output),
        raw=args.raw,
        time_series_interval=None,
        no_delete_on_exit=False,
        _quiet=True,
        _user_namespace=namespace,
        _sweep_context={
            "type": "memory-growth",
            "memory_count": memory_count,
            "repetition": repetition,
            "queries_per_user": args.queries_per_user,
        },
    )


def _validate_args(
    args: argparse.Namespace,
) -> tuple[list[int], BenchmarkConfig]:
    counts = parse_memory_counts(args.memory_counts)
    if args.duration <= 0 and args.ops <= 0:
        raise ValueError("memory-growth sweep requires either --duration or --ops")
    if args.users <= 0:
        raise ValueError("--users must be > 0")
    if args.procs <= 0 or args.procs > args.users:
        raise ValueError("--procs must be between 1 and --users")
    if args.global_concurrency < 0:
        raise ValueError("--global-concurrency must be >= 0")
    if args.warmup < 0 or args.rampup < 0:
        raise ValueError("--warmup and --rampup must be >= 0")
    if args.queries_per_user <= 0:
        raise ValueError("--queries-per-user must be > 0")
    if args.queries_per_user > counts[0]:
        raise ValueError("--queries-per-user cannot exceed the smallest memory count")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be > 0")
    if not 0.0 <= args.max_empty_rate <= 1.0:
        raise ValueError("--max-empty-rate must be between 0 and 1")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.expand < 0:
        raise ValueError("--expand must be >= 0")
    server_metrics_interval = getattr(args, "server_metrics_interval", None)
    if server_metrics_interval is not None:
        if not math.isfinite(server_metrics_interval) or server_metrics_interval <= 0:
            raise ValueError("--server-metrics-interval must be a finite number > 0")
        if not args.server_metrics:
            raise ValueError("--server-metrics-interval requires --server-metrics")
    cfg = load_config(args.config)
    if cfg.backend.options.get("project_id"):
        raise ValueError(
            "memory-growth sweep requires per-user backend projects; a fixed "
            "backend.project_id cannot isolate corpus-size points"
        )
    return counts, cfg


def _validate_dataset_capacity(
    args: argparse.Namespace,
    cfg: BenchmarkConfig,
    required: int,
) -> None:
    """Fail before backend work when any selected user is too short.

    Dataset streams may vary by user, so checking only the first user would
    make the largest point conditional on assignment order. Consumption is
    bounded at ``required`` and does not materialize the items.
    """
    dataset = build_dataset(cfg.dataset)
    users = dataset.users(args.users, seed=args.seed)
    if len(users) != args.users:
        raise ValueError(
            f"dataset returned {len(users)} users, but the sweep requires {args.users}"
        )
    for user in users:
        available = sum(1 for _ in islice(dataset.memory_stream(user), required))
        if available < required:
            raise ValueError(
                f"largest memory point {required} requires at least {required} "
                f"memory items for user {user!r}, but the dataset yielded "
                f"{available}"
            )


def run_memory_growth(
    args: argparse.Namespace,
    run_one: Callable[[argparse.Namespace], int],
) -> int:
    """Run isolated, repeated ``search-load`` points as corpus size grows."""
    counts, cfg = _validate_args(args)
    output = Path(args.output)
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    _validate_dataset_capacity(args, cfg, max(counts))
    output.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    sweep_id = uuid4().hex[:12]
    manifest: dict[str, Any] = {
        "version": 1,
        "type": "memory-growth",
        "status": "running",
        "started_at": started_at.isoformat(),
        "ended_at": None,
        "config": {
            "config": args.config,
            "memory_counts": counts,
            "queries_per_user": args.queries_per_user,
            "users": args.users,
            "seed": args.seed,
            "duration": args.duration,
            "ops": args.ops,
            "warmup": args.warmup,
            "rampup": args.rampup,
            "global_concurrency": args.global_concurrency,
            "procs": args.procs,
            "top_k": args.top_k,
            "expand_context": args.expand,
            "filter": args.filter,
            "repetitions": args.repetitions,
            "max_empty_rate": args.max_empty_rate,
            "server_metrics": args.server_metrics,
            "server_metrics_interval": getattr(args, "server_metrics_interval", None),
            "raw": args.raw,
        },
        "points": [],
    }
    generation = dataset_settings(cfg.dataset, seed=args.seed)
    if generation is not None:
        manifest["config"]["dataset_config"] = generation
    write_sweep_reports(manifest, output)

    for memory_count in counts:
        point: dict[str, Any] = {
            "memory_count": memory_count,
            "status": "running",
            "repetitions": [],
            "aggregate": {},
        }
        manifest["points"].append(point)
        for repetition in range(1, args.repetitions + 1):
            relative = Path(f"n_{memory_count}") / f"repeat_{repetition}"
            namespace = f"mg_{sweep_id}_n{memory_count}_r{repetition}_"
            repeat: dict[str, Any] = {
                "repetition": repetition,
                "status": "running",
                "output": relative.as_posix(),
                "namespace": namespace,
                "metrics": {},
                "reasons": [],
            }
            point["repetitions"].append(repeat)
            run_args = _run_args(
                args,
                memory_count=memory_count,
                repetition=repetition,
                namespace=namespace,
                output=output / relative,
            )
            try:
                result = run_one(run_args)
                if result:
                    raise RuntimeError(f"benchmark run exited with status {result}")
                with open(output / relative / "summary.json", encoding="utf-8") as f:
                    payload = json.load(f)
                metrics, reasons = evaluate_repeat(
                    payload,
                    memory_count=memory_count,
                    users=args.users,
                    max_empty_rate=args.max_empty_rate,
                )
                repeat["metrics"] = metrics
                repeat["reasons"] = reasons
                repeat["status"] = "invalid" if reasons else "ok"
            except Exception as error:
                logger.exception(
                    "memory-growth point %s repetition %s failed",
                    memory_count,
                    repetition,
                )
                repeat["status"] = "failed"
                repeat["error"] = {
                    "kind": type(error).__name__,
                    "message": str(error),
                }
            point["aggregate"] = aggregate_repeats(point["repetitions"])
            manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
            write_sweep_reports(manifest, output)

        statuses = {repeat["status"] for repeat in point["repetitions"]}
        point["status"] = _combined_status(statuses)
        write_sweep_reports(manifest, output)

    statuses = {point["status"] for point in manifest["points"]}
    manifest["status"] = _combined_status(statuses)
    manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
    write_sweep_reports(manifest, output)
    print(json.dumps(manifest, indent=2))
    print(f"memory-growth sweep written to {output}/")
    return 0 if manifest["status"] == "ok" else 1


def _combined_status(statuses: set[str]) -> str:
    if "failed" in statuses:
        return "failed"
    if "invalid" in statuses:
        return "invalid"
    return "ok"


__all__ = [
    "NamespacedDataset",
    "aggregate_repeats",
    "evaluate_repeat",
    "parse_memory_counts",
    "run_memory_growth",
    "write_sweep_reports",
]
