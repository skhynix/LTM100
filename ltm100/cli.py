"""LTM100 command-line interface.

Top-level commands:
  run      Drive a benchmark run: provision users, run a scenario, report.
  sweep    Repeat a controlled experiment across parameter points.
  cleanup  Delete per-user state for a run (without running).

Per-run parameters come from CLI flags; adapter choices and endpoint/auth
come from the YAML config file.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import math
import sys
from datetime import datetime, timezone
from typing import Any

from ltm100.adapters.backends.memory_types import memory_type_settings
from ltm100.config import build_backend, build_dataset, load_config
from ltm100.core.chat_profile import load_chat_profile
from ltm100.core.config import RunConfig
from ltm100.core.memory_growth import (
    NamespacedDataset,
    run_memory_growth,
)
from ltm100.core.multiproc import ShardResult, run_shards
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import get_scenario
from ltm100.metrics.aggregate import aggregate
from ltm100.metrics.report import (
    write_raw_ndjson,
    write_server_metrics,
    write_server_metrics_timeseries,
    write_summary_csv,
    write_summary_json,
    write_timeseries_csv,
)
from ltm100.metrics.server_metrics import SnapshotCollector, finish, finish_timeseries
from ltm100.metrics.timeseries import aggregate_timeseries

logger = logging.getLogger(__name__)

# Set by _run before spawning when --server-metrics resolved for the
# single-process case: the one shard IS this process, so its runner can call
# the collector's snapshots as measure hooks. Spawned children start with a
# fresh import of this module, so the collector never reaches them (their
# window is covered by the parent's whole-run scrape instead).
_server_metrics_collector: SnapshotCollector | None = None


def _split(total: int, procs: int, index: int) -> int:
    """This shard's share of a whole-run integer budget.

    The remainder goes to the lowest-numbered shards, so the shares sum to the
    total exactly rather than each shard rounding up.
    """
    if procs <= 1 or total <= 0:
        return total
    return total // procs + (1 if index < total % procs else 0)


def _build_run_config(args: argparse.Namespace) -> RunConfig:
    # Every field below that describes the WHOLE run's load has to be divided
    # across shards. Each shard builds its own runner, so an undivided value is
    # applied N times over: --procs 4 --global-concurrency 100 would cap at 400
    # in flight, and an open-model arrival rate would fire N times too fast.
    # users is not here because shard_users already partitions it, and
    # session_ops is per-session rather than per-run.
    procs = args.procs
    index = getattr(args, "proc_index", 0)
    return RunConfig(
        users=args.users,
        seed=args.seed,
        duration=args.duration,
        ops=_split(args.ops, procs, index),
        global_concurrency=_split(args.global_concurrency, procs, index),
        warmup=args.warmup,
        rampup=args.rampup,
        preingest=args.preingest,
        preingest_fraction=args.preingest_fraction,
        preingest_items_per_user=args.preingest_items_per_user,
        model=args.model,
        arrival_rate=(args.arrival_rate / procs if procs > 1 else args.arrival_rate),
        session_ops=args.session_ops,
        queue_bound=_split(args.queue_bound, procs, index),
        delete_on_exit=not args.no_delete_on_exit,
        procs=args.procs,
        proc_index=getattr(args, "proc_index", 0),
    )


def _build_scenario(args: argparse.Namespace):
    if args.chat_profile and args.scenario != "chat-replay":
        raise ValueError("--chat-profile is only valid with --scenario chat-replay")
    if args.query_limit is not None and args.scenario != "search-load":
        raise ValueError("--query-limit is only valid with --scenario search-load")
    kwargs: dict[str, Any] = {}
    if args.scenario == "mixed":
        kwargs["search_weight"] = args.search_weight
        kwargs["think"] = args.think
        kwargs["top_k"] = args.top_k
    elif args.scenario == "chat-replay":
        kwargs["think"] = args.think
        kwargs["search_every"] = args.search_every
        kwargs["answer_time"] = args.answer_time
        kwargs["user_gap"] = args.user_gap
        kwargs["answer_time_variation"] = args.answer_time_variation
        kwargs["user_gap_variation"] = args.user_gap_variation
        kwargs["top_k"] = args.top_k
        if args.chat_profile:
            kwargs["profile"] = load_chat_profile(
                args.chat_profile,
                think=args.think,
                search_every=args.search_every,
                answer_time=args.answer_time,
                user_gap=args.user_gap,
                top_k=args.top_k,
                answer_time_variation=args.answer_time_variation,
                user_gap_variation=args.user_gap_variation,
            )
    elif args.scenario == "search-load":
        kwargs["top_k"] = args.top_k
        kwargs["query_limit"] = args.query_limit
    # Every scenario that searches takes the server-side search knobs.
    if args.scenario in ("mixed", "chat-replay", "search-load"):
        kwargs["expand_context"] = args.expand
        kwargs["filter"] = args.filter
    return get_scenario(args.scenario, **kwargs)


async def _run_shard_async(args: argparse.Namespace) -> ShardResult:
    cfg = load_config(args.config)
    dataset = build_dataset(cfg.dataset)
    namespace = getattr(args, "_user_namespace", "")
    if namespace:
        dataset = NamespacedDataset(dataset, namespace)
    backend = build_backend(cfg.backend)
    run_cfg = _build_run_config(args)
    scenario = _build_scenario(args)

    hooks: dict[str, Any] = {}
    measure_sync = getattr(args, "_measure_sync", None)
    if measure_sync is not None:

        async def synchronized_start() -> None:
            await asyncio.to_thread(
                measure_sync["queue"].put,
                ("start", args.proc_index, ""),
            )
            await asyncio.to_thread(measure_sync["start_release"].wait)

        async def synchronized_end() -> None:
            await asyncio.to_thread(
                measure_sync["queue"].put,
                ("end", args.proc_index, ""),
            )
            await asyncio.to_thread(measure_sync["end_release"].wait)

        hooks = {
            "on_measure_start": synchronized_start,
            "on_measure_end": synchronized_end,
        }
    elif _server_metrics_collector is not None:
        # procs == 1 only (see the module-level note): this shard is the
        # process the flag was set in, so its measured window is exactly the
        # window the user asked about -- pre-ingest and teardown excluded.
        hooks = {
            "on_measure_start": _server_metrics_collector.start,
            "on_measure_end": _server_metrics_collector.end,
        }
    runner = LoadRunner(
        client=backend,
        dataset=dataset,
        scenario=scenario,
        config=run_cfg,
        **hooks,
    )
    async with backend:  # type: ignore[arg-type]
        preflight = getattr(backend, "preflight", None)
        if preflight is not None:
            await preflight()
        run_failed = False
        try:
            await runner.run()
        except BaseException:
            run_failed = True
            raise
        finally:
            users = getattr(runner, "users", [])
            if run_cfg.delete_on_exit and users:
                try:
                    # Each shard owns the users it drove, so it tears down its own.
                    await backend.teardown(users, delete=True)
                except Exception:
                    if not run_failed:
                        raise
                    # Preserve the workload failure that triggered cleanup;
                    # teardown is best-effort only on that exceptional path.
                    logger.warning(
                        "teardown failed after the benchmark run failed",
                        exc_info=True,
                    )
    return ShardResult(
        runner.recorder.raw(),
        runner.session_stats,
        runner.preingest_stats,
        runner.measurement_started_at,
        runner.measurement_ended_at,
    )


def _shard_entry(args_dict: dict, proc_index: int) -> list:
    """Entry point for a spawned shard; must be importable by name."""
    args = argparse.Namespace(**args_dict)
    args.proc_index = proc_index
    if proc_index > 0:
        logging.basicConfig(level=logging.WARNING)
    try:
        return asyncio.run(_run_shard_async(args))
    except BaseException as error:
        measure_sync = getattr(args, "_measure_sync", None)
        if measure_sync is not None:
            measure_sync["queue"].put(
                ("error", proc_index, f"{type(error).__name__}: {error}")
            )
            measure_sync["start_release"].set()
            measure_sync["end_release"].set()
        raise


def _backend_build(cfg) -> dict:
    """What the server under test reports about itself.

    A throughput number is not reproducible without the build that produced it,
    and the version is the one thing the harness cannot infer: the same tag can
    be rebuilt, and a locally built image often reports 0.0.0 precisely because
    nothing stamped it. Asked once, before the run, over its own connection.
    """

    async def probe() -> dict:
        backend = build_backend(cfg.backend)
        health = getattr(backend, "health", None)
        if health is None:
            return {}
        async with backend:  # type: ignore[arg-type]
            return await health()

    try:
        reported = asyncio.run(probe())
    except Exception as e:  # noqa: BLE001 - a failed probe must not cost the run
        return {"build": f"<unavailable: {type(e).__name__}>"}
    if not reported:
        return {}
    return {
        "build": reported.get("version") or "<unreported>",
        "service": reported.get("service") or "<unreported>",
    }


def _run_metadata(
    args: argparse.Namespace,
    *,
    dataset: str,
    backend: str,
    build: dict,
    started_at: datetime,
    ended_at: datetime,
    preingest_stats: dict[str, int | None] | None = None,
    measurement_started_at: float | None = None,
    measurement_ended_at: float | None = None,
) -> dict:
    """Describe the whole run, never an individual process shard."""
    meta = {
        "dataset": dataset,
        "backend": backend,
        **build,
        "scenario": args.scenario,
        "users": args.users,
        "seed": args.seed,
        "duration": args.duration,
        "ops": args.ops,
        "global_concurrency": args.global_concurrency,
        "warmup": args.warmup,
        "rampup": args.rampup,
        "preingest": args.preingest,
        "preingest_fraction": (
            args.preingest_fraction if args.preingest_items_per_user is None else None
        ),
        "preingest_items_per_user": args.preingest_items_per_user,
        "model": args.model,
        "procs": args.procs,
        "delete_on_exit": not args.no_delete_on_exit,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
    }
    if measurement_started_at is not None and measurement_ended_at is not None:
        meta.update(
            measurement_started_at=datetime.fromtimestamp(
                measurement_started_at, timezone.utc
            ).isoformat(),
            measurement_ended_at=datetime.fromtimestamp(
                measurement_ended_at, timezone.utc
            ).isoformat(),
        )
    time_series_interval = getattr(args, "time_series_interval", None)
    if time_series_interval is not None:
        meta["time_series_interval"] = time_series_interval
    server_metrics_interval = getattr(args, "server_metrics_interval", None)
    if server_metrics_interval is not None:
        meta["server_metrics_interval"] = server_metrics_interval

    if args.model == "open":
        meta.update(
            arrival_rate=args.arrival_rate,
            session_ops=args.session_ops,
            queue_bound=args.queue_bound,
        )
    if args.scenario in ("search-load", "mixed", "chat-replay"):
        meta.update(
            top_k=args.top_k,
            expand_context=args.expand,
            filter=args.filter,
        )
    if args.scenario == "search-load":
        meta["query_limit"] = args.query_limit
    if args.preingest and preingest_stats is not None:
        meta["preingest_stats"] = preingest_stats
    if args.scenario == "mixed":
        meta.update(search_weight=args.search_weight, think=args.think)
    elif args.scenario == "chat-replay":
        meta.update(
            think=args.think,
            search_every=args.search_every,
            answer_time=args.answer_time,
            user_gap=args.user_gap,
            answer_time_variation=args.answer_time_variation,
            user_gap_variation=args.user_gap_variation,
        )
        if args.chat_profile:
            profile = load_chat_profile(
                args.chat_profile,
                think=args.think,
                search_every=args.search_every,
                answer_time=args.answer_time,
                user_gap=args.user_gap,
                top_k=args.top_k,
                answer_time_variation=args.answer_time_variation,
                user_gap_variation=args.user_gap_variation,
            )
            meta["chat_profile"] = profile.metadata(args.users)

    sweep = getattr(args, "_sweep_context", None)
    if sweep is not None:
        meta["sweep"] = sweep

    return meta


def _probe_server_metrics(cfg) -> dict:
    """Resolve --server-metrics against the backend before the run starts.

    Three answers, none of them fatal, keyed by one discriminating field:
      - {"unsupported": ...} -- the backend class does not declare the
        capability; not probed, nothing to probe against;
      - {"failed": reason}   -- it declares the capability but the endpoint
        errored or answered empty (older server build, wrong URL);
      - {"enabled": True}    -- it declares it and GET /api/v2/metrics answered.
    """

    async def probe() -> dict:
        backend = build_backend(cfg.backend)
        if not getattr(backend, "supports_server_metrics", False):
            return {"unsupported": True}
        async with backend:  # type: ignore[arg-type]
            text = await backend.server_metrics_snapshot()
        if not isinstance(text, str) or not text.strip():
            return {"failed": "empty response from the metrics endpoint"}
        return {"enabled": True}

    try:
        return asyncio.run(probe())
    except Exception as e:  # noqa: BLE001 - never cost the run
        return {"failed": f"{type(e).__name__}: {e}"}


def _run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    run_cfg = _build_run_config(args)
    time_series_interval = getattr(args, "time_series_interval", None)
    if time_series_interval is not None:
        if not math.isfinite(time_series_interval) or time_series_interval <= 0:
            raise ValueError("--time-series-interval must be a finite number > 0")
        if not args.output:
            raise ValueError("--time-series-interval requires --output")
    server_metrics_interval = getattr(args, "server_metrics_interval", None)
    if server_metrics_interval is not None:
        if not math.isfinite(server_metrics_interval) or server_metrics_interval <= 0:
            raise ValueError("--server-metrics-interval must be a finite number > 0")
        if not args.server_metrics:
            raise ValueError("--server-metrics-interval requires --server-metrics")
        if not args.output:
            raise ValueError("--server-metrics-interval requires --output")
    # Validate scenario-specific files and options before probing or mutating
    # the backend. Each shard rebuilds the scenario it will actually run.
    _build_scenario(args)
    type_settings = memory_type_settings(cfg.backend.name, cfg.backend.options)
    if (
        type_settings
        and "semantic" in type_settings["resolved_request_types"]["search"]
        and (args.expand or args.filter)
    ):
        raise ValueError("semantic search does not support --expand or --filter")
    # Each shard tears down the users it drove, which assumes a project per
    # user. With backend.project_id set they all share one, so the first shard
    # to finish would delete it under the others mid-run. Refuse rather than
    # coordinate: an isolation-scope arm wants the corpus kept anyway.
    if (
        run_cfg.procs > 1
        and run_cfg.delete_on_exit
        and cfg.backend.options.get("project_id")
    ):
        raise ValueError(
            "backend.project_id puts every user in one project, so --procs > 1 "
            "cannot delete on exit: whichever shard finishes first would drop "
            "the project the others are still using. Pass --no-delete-on-exit "
            "and remove the project yourself, or run with --procs 1."
        )
    # A shared project with no producer filter is a legitimate shape -- one
    # memory pool everyone searches -- but it is easily mistaken for the
    # isolation measurement, so say which one this run is.
    options = cfg.backend.options
    if options.get("project_id") and not options.get("filter_by_producer"):
        logger.warning(
            "backend.project_id is set without filter_by_producer: every virtual "
            "user searches every other user's memories. Set filter_by_producer: "
            "true for per-user isolation inside the shared project."
        )
    # Before the run: a server that dies under load still has to be identifiable.
    build = _backend_build(cfg)

    # --server-metrics: resolve once, before any load, so the warning a
    # user gets describes the backend they configured rather than a
    # mid-run surprise.
    collector: SnapshotCollector | None = None
    server_metrics: dict | None = None
    if args.server_metrics:
        resolution = _probe_server_metrics(cfg)
        if resolution.get("unsupported"):
            logger.warning(
                "--server-metrics: backend adapter %r does not implement a "
                "server metrics query; disabled for this run",
                cfg.backend.name,
            )
            server_metrics = {
                "enabled": False,
                "status": "unsupported",
                "window": None,
                "warnings": [
                    (
                        f"backend adapter {cfg.backend.name!r} implements no server "
                        "metrics query"
                    ),
                ],
                "rows": [],
            }
        elif resolution.get("failed"):
            logger.warning(
                "--server-metrics: endpoint probe failed (%s); disabled for this run",
                resolution["failed"],
            )
            server_metrics = {
                "enabled": False,
                "status": "failed",
                "window": None,
                "warnings": [f"metrics endpoint probe failed: {resolution['failed']}"],
                "rows": [],
            }
        elif run_cfg.procs == 1:
            # The single shard runs in this process, so its runner's measure
            # hooks bracket the measured window exactly.
            collector = SnapshotCollector(build_backend(cfg.backend))
        else:
            collector = SnapshotCollector(build_backend(cfg.backend))
        if collector is not None and server_metrics_interval is not None:
            collector.interval_seconds = server_metrics_interval

    global _server_metrics_collector
    _server_metrics_collector = collector if run_cfg.procs == 1 else None
    try:
        started_at = datetime.now(timezone.utc)
        synchronize_workers = run_cfg.procs > 1 and (
            collector is not None
            or run_cfg.warmup > 0
            or time_series_interval is not None
        )
        if synchronize_workers:
            shard_result = run_shards(
                _shard_entry,
                vars(args),
                run_cfg.procs,
                on_measure_start=(
                    (lambda: asyncio.run(collector.start()))
                    if collector is not None
                    else (lambda: None)
                ),
                on_measure_end=(
                    (lambda: asyncio.run(collector.end()))
                    if collector is not None
                    else (lambda: None)
                ),
            )
        else:
            shard_result = run_shards(_shard_entry, vars(args), run_cfg.procs)
        ended_at = datetime.now(timezone.utc)
    finally:
        _server_metrics_collector = None

    server_metrics_timeseries: dict[str, Any] | None = None
    if collector is not None:
        capture = collector.result()
        server_metrics = finish(capture, window="measured")
        if server_metrics_interval is not None:
            server_metrics_timeseries = finish_timeseries(capture)
            server_metrics["timeseries"] = {
                key: value
                for key, value in server_metrics_timeseries.items()
                if key != "rows"
            }
    elif server_metrics_interval is not None:
        server_metrics_timeseries = {
            "enabled": False,
            "status": server_metrics.get("status", "failed")
            if server_metrics
            else "failed",
            "interval_seconds": server_metrics_interval,
            "intervals": 0,
            "valid_intervals": 0,
            "failed_scrapes": 0,
            "warnings": list(server_metrics.get("warnings", []))
            if server_metrics
            else [],
            "rows": [],
        }
        if server_metrics is not None:
            server_metrics["timeseries"] = {
                key: value
                for key, value in server_metrics_timeseries.items()
                if key != "rows"
            }

    if isinstance(shard_result, ShardResult):
        raw = shard_result.results
        session_stats = shard_result.sessions
        preingest_stats = shard_result.preingest
        measurement_started_at = shard_result.measurement_started_at
        measurement_ended_at = shard_result.measurement_ended_at
    else:  # compatibility with integrations that wrap run_shards
        raw = shard_result
        session_stats = None
        preingest_stats = None
        measurement_started_at = min(
            (result.started_at for result in raw), default=None
        )
        measurement_ended_at = max((result.ended_at for result in raw), default=None)
    summary = aggregate(raw)
    if args.model == "open" and session_stats is not None:
        summary["sessions"] = session_stats.as_dict()

    meta = _run_metadata(
        args,
        dataset=cfg.dataset.name,
        backend=cfg.backend.name,
        build=build,
        started_at=started_at,
        ended_at=ended_at,
        preingest_stats=(
            preingest_stats.as_dict() if preingest_stats is not None else None
        ),
        measurement_started_at=measurement_started_at,
        measurement_ended_at=measurement_ended_at,
    )

    if type_settings is not None:
        meta["memory_types"] = type_settings

    # The section's `raw` block is the full two-snapshot scrape: too big for
    # the inline copy, which exists to be read, and goes to its own file.
    display_metrics = None
    if server_metrics is not None:
        display_metrics = {k: v for k, v in server_metrics.items() if k != "raw"}
    payload: dict[str, Any] = {"meta": meta, "summary": summary}
    if display_metrics is not None:
        payload["server_metrics"] = display_metrics
    if not getattr(args, "_quiet", False):
        print(json.dumps(payload, indent=2))

    if args.output:
        out = args.output.rstrip("/")
        write_summary_json(
            summary, f"{out}/summary.json", meta=meta, server_metrics=display_metrics
        )
        write_summary_csv(summary, f"{out}/summary.csv")
        if server_metrics is not None:
            write_server_metrics(server_metrics, out)
        if server_metrics_timeseries is not None:
            write_server_metrics_timeseries(
                server_metrics_timeseries["rows"],
                f"{out}/server_metrics_timeseries.csv",
            )
        if args.raw:
            write_raw_ndjson(raw, f"{out}/raw.ndjson")
        if time_series_interval is not None:
            if measurement_started_at is None or measurement_ended_at is None:
                raise RuntimeError(
                    "measured window is unavailable for time-series output"
                )
            rows = aggregate_timeseries(
                raw,
                interval_seconds=time_series_interval,
                measurement_started_at=measurement_started_at,
                measurement_ended_at=measurement_ended_at,
            )
            write_timeseries_csv(rows, f"{out}/timeseries.csv")
        if not getattr(args, "_quiet", False):
            print(f"reports written to {out}/")

    return 0


def _run_memory_growth(args: argparse.Namespace) -> int:
    return run_memory_growth(args, _run)


async def _cleanup(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    dataset = build_dataset(cfg.dataset)
    backend = build_backend(cfg.backend)
    users = dataset.users(args.users, seed=args.seed)
    async with backend:  # type: ignore[arg-type]
        await backend.teardown(users, delete=True)
    print(f"deleted state for {len(users)} users")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ltm100", description="LTM100 load benchmark."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, help="path to YAML config")
    common.add_argument("--users", type=int, default=10, help="virtual users")
    common.add_argument("--seed", type=int, default=0, help="RNG seed")

    run = sub.add_parser("run", parents=[common], help="run a benchmark")
    run.add_argument("--scenario", required=True, help="scenario name")
    g = run.add_mutually_exclusive_group()
    g.add_argument(
        "--duration", type=float, default=0.0, help="measured run seconds (0=off)"
    )
    g.add_argument("--ops", type=int, default=0, help="measured total ops cap (0=off)")
    run.add_argument("--global-concurrency", type=int, default=0, help="max in-flight")
    run.add_argument(
        "--warmup",
        type=float,
        default=0.0,
        help="unmeasured workload seconds before the measured run",
    )
    run.add_argument(
        "--preingest", action="store_true", help="pre-ingest memories before run"
    )
    preingest_size = run.add_mutually_exclusive_group()
    preingest_size.add_argument(
        "--preingest-fraction",
        type=float,
        default=1.0,
        help="fraction of each user's memories to pre-ingest",
    )
    preingest_size.add_argument(
        "--preingest-items-per-user",
        type=int,
        default=None,
        help="exact input items to pre-ingest per user; streams only the "
        "requested prefix and is mutually exclusive with "
        "--preingest-fraction",
    )
    run.add_argument("--rampup", type=float, default=0.0, help="ramp-up seconds")
    run.add_argument(
        "--model",
        choices=("closed", "open"),
        default="closed",
        help="load model (closed=fixed N looping users; open=Poisson arrivals)",
    )
    run.add_argument(
        "--arrival-rate",
        type=float,
        default=0.0,
        help="open model: user arrivals per second (Poisson lambda)",
    )
    run.add_argument(
        "--session-ops",
        type=int,
        default=0,
        help="open model: ops each arriving user performs before leaving",
    )
    run.add_argument(
        "--queue-bound",
        type=int,
        default=0,
        help="open model: max queued beyond cap before rejection (0=reject on cap)",
    )
    run.add_argument(
        "--search-weight",
        type=float,
        default=0.8,
        help="mixed scenario: fraction of ops that are search (0..1)",
    )
    run.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="search top_k: how many memories the backend returns per search "
        "(search-load, mixed, chat-replay; default 20). Applied uniformly to "
        "all users",
    )
    run.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="search-load: build each user's query pool from exactly the "
        "first N dataset memories (default: entire memory stream)",
    )
    run.add_argument(
        "--think",
        type=float,
        default=0.05,
        help="mixed/chat-replay: max think-time jitter per op (seconds)",
    )
    run.add_argument(
        "--search-every",
        type=int,
        default=1,
        help="chat-replay: issue a recall search every N user turns (default 1 = every user turn)",
    )
    run.add_argument(
        "--chat-profile",
        default=None,
        help="chat-replay: YAML user-group workload profile; group values "
        "override the corresponding CLI defaults",
    )
    run.add_argument(
        "--answer-time",
        type=float,
        default=0.0,
        help="chat-replay: mean seconds the LLM spends generating an answer "
        "after a user turn (Exponential; 0 = back-to-back, default). Applied "
        "uniformly to all users",
    )
    run.add_argument(
        "--answer-time-variation",
        type=float,
        default=None,
        help="chat-replay: optional bounded variation ratio around --answer-time "
        "(0 = fixed, 0.3 = +/-30%%, 1 = 0..2x). When omitted, preserves the "
        "legacy Exponential distribution",
    )
    run.add_argument(
        "--user-gap",
        type=float,
        default=0.0,
        help="chat-replay: mean seconds the user takes before the next turn "
        "(Exponential; 0 = back-to-back, default). Applied uniformly to all users",
    )
    run.add_argument(
        "--user-gap-variation",
        type=float,
        default=None,
        help="chat-replay: optional bounded variation ratio around --user-gap "
        "(0 = fixed, 0.5 = +/-50%%, 1 = 0..2x). When omitted, preserves the "
        "legacy Exponential distribution",
    )
    run.add_argument(
        "--expand",
        type=int,
        default=0,
        help="search: expand_context, the number of neighbouring episodes the "
        "server returns around each hit (default 0 = off, and the field is then "
        "omitted from the request)",
    )
    run.add_argument(
        "--filter",
        default="",
        help="search: a server-side metadata filter, e.g. "
        "'metadata.category=cat_3'. Exact match, no quoting. Requires the corpus "
        "to carry that field - the synthetic dataset writes metadata.category "
        "when its `categories` option is set",
    )
    run.add_argument(
        "--procs",
        type=int,
        default=1,
        help="OS processes to shard virtual users across (default 1). One "
        "asyncio process saturates a core well before the server does, so "
        "large user counts need several. --procs 1 is single-process.",
    )
    run.add_argument(
        "--server-metrics",
        action="store_true",
        help="scrape the server's own Prometheus latency metrics around the "
        "run and report per-phase deltas (implemented for the MemMachine "
        "REST adapter; adapters without the query warn and continue "
        "without it)",
    )
    run.add_argument(
        "--server-metrics-interval",
        type=float,
        default=None,
        help="with --server-metrics, write server_metrics_timeseries.csv from "
        "Prometheus histogram deltas sampled every N measured seconds",
    )
    run.add_argument("--output", default=None, help="output dir for reports")
    run.add_argument("--raw", action="store_true", help="also write raw.ndjson")
    run.add_argument(
        "--time-series-interval",
        type=float,
        default=None,
        help="write timeseries.csv with client-observed E2E metrics aggregated "
        "into fixed intervals of this many seconds",
    )
    run.add_argument("--no-delete-on-exit", action="store_true", help="keep user state")
    run.set_defaults(func=_run)

    sweep = sub.add_parser("sweep", help="run a controlled experiment sweep")
    sweep_sub = sweep.add_subparsers(dest="sweep", required=True)
    growth = sweep_sub.add_parser(
        "memory-growth",
        parents=[common],
        help="measure search behavior as stored memories per user grow",
    )
    growth.add_argument(
        "--memory-counts",
        required=True,
        help="strictly increasing memories-per-user points, e.g. 100,1000,10000",
    )
    growth.add_argument(
        "--queries-per-user",
        type=int,
        required=True,
        help="fixed source-memory prefix used as each user's query pool",
    )
    growth_termination = growth.add_mutually_exclusive_group(required=True)
    growth_termination.add_argument(
        "--duration", type=float, default=0.0, help="measured seconds per repetition"
    )
    growth_termination.add_argument(
        "--ops", type=int, default=0, help="measured total search ops per repetition"
    )
    growth.add_argument(
        "--repetitions", type=int, default=3, help="repetitions per memory point"
    )
    growth.add_argument(
        "--max-empty-rate",
        type=float,
        default=0.0,
        help="largest acceptable successful-search empty rate (default 0)",
    )
    growth.add_argument(
        "--global-concurrency", type=int, default=0, help="max in-flight searches"
    )
    growth.add_argument(
        "--warmup",
        type=float,
        default=0.0,
        help="unmeasured workload seconds before each measured repetition",
    )
    growth.add_argument("--rampup", type=float, default=0.0, help="ramp-up seconds")
    growth.add_argument(
        "--top-k", type=int, default=20, help="results requested per search"
    )
    growth.add_argument(
        "--expand", type=int, default=0, help="server-side context expansion"
    )
    growth.add_argument(
        "--filter", default="", help="server-side metadata filter expression"
    )
    growth.add_argument(
        "--procs", type=int, default=1, help="load-generator process count"
    )
    growth.add_argument(
        "--server-metrics",
        action="store_true",
        help="collect backend-provided server metrics for every repetition",
    )
    growth.add_argument(
        "--server-metrics-interval",
        type=float,
        default=None,
        help="with --server-metrics, collect fixed-interval server latency "
        "breakdowns for every repetition",
    )
    growth.add_argument(
        "--raw", action="store_true", help="write raw.ndjson for every repetition"
    )
    growth.add_argument(
        "--output", required=True, help="empty output directory for sweep reports"
    )
    growth.set_defaults(func=_run_memory_growth)

    clean = sub.add_parser("cleanup", parents=[common], help="delete per-user state")
    clean.set_defaults(func=_cleanup)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if (
        not getattr(args, "duration", 0)
        and not getattr(args, "ops", 0)
        and args.command == "run"
    ):
        parser.error("run requires either --duration or --ops")
    # The run path owns its own event loops (one per shard process), so only
    # the coroutine commands are wrapped here.
    result = args.func(args)
    if inspect.iscoroutine(result):
        return asyncio.run(result)
    return result


if __name__ == "__main__":
    sys.exit(main())
