"""LTM100 command-line interface.

Top-level commands:
  run      Drive a benchmark run: provision users, run a scenario, report.
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
import sys
from datetime import datetime, timezone
from typing import Any

from ltm100.config import build_backend, build_dataset, load_config
from ltm100.core.config import RunConfig
from ltm100.core.multiproc import run_shards
from ltm100.core.runner import LoadRunner
from ltm100.core.scenarios import get_scenario
from ltm100.metrics.aggregate import aggregate
from ltm100.metrics.report import (
    write_raw_ndjson,
    write_server_metrics,
    write_summary_csv,
    write_summary_json,
)
from ltm100.metrics.server_metrics import SnapshotCollector, finish

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
        model=args.model,
        arrival_rate=(args.arrival_rate / procs if procs > 1 else args.arrival_rate),
        session_ops=args.session_ops,
        queue_bound=_split(args.queue_bound, procs, index),
        delete_on_exit=not args.no_delete_on_exit,
        procs=args.procs,
        proc_index=getattr(args, "proc_index", 0),
    )


def _build_scenario(args: argparse.Namespace):
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
        kwargs["top_k"] = args.top_k
    elif args.scenario == "search-load":
        kwargs["top_k"] = args.top_k
    # Every scenario that searches takes the server-side search knobs.
    if args.scenario in ("mixed", "chat-replay", "search-load"):
        kwargs["expand_context"] = args.expand
        kwargs["filter"] = args.filter
    return get_scenario(args.scenario, **kwargs)


async def _run_shard_async(args: argparse.Namespace) -> list:
    cfg = load_config(args.config)
    dataset = build_dataset(cfg.dataset)
    backend = build_backend(cfg.backend)
    run_cfg = _build_run_config(args)
    scenario = _build_scenario(args)

    hooks: dict[str, Any] = {}
    if _server_metrics_collector is not None:
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
        await runner.run()
        if run_cfg.delete_on_exit:
            # Each shard owns the users it drove, so it tears down its own.
            await backend.teardown(runner.users, delete=True)
    return runner.recorder.raw()


def _shard_entry(args_dict: dict, proc_index: int) -> list:
    """Entry point for a spawned shard; must be importable by name."""
    args = argparse.Namespace(**args_dict)
    args.proc_index = proc_index
    if proc_index > 0:
        logging.basicConfig(level=logging.WARNING)
    return asyncio.run(_run_shard_async(args))


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
        "preingest_fraction": args.preingest_fraction,
        "model": args.model,
        "procs": args.procs,
        "delete_on_exit": not args.no_delete_on_exit,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
    }

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
    if args.scenario == "mixed":
        meta.update(search_weight=args.search_weight, think=args.think)
    elif args.scenario == "chat-replay":
        meta.update(
            think=args.think,
            search_every=args.search_every,
            answer_time=args.answer_time,
            user_gap=args.user_gap,
        )

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
                    f"backend adapter {cfg.backend.name!r} implements no server "
                    "metrics query"
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
            logger.warning(
                "--server-metrics with --procs %d: the parent cannot see inside "
                "the shards' measured windows, so the snapshots bracket the "
                "whole run including setup and pre-ingest (window=whole_run)",
                run_cfg.procs,
            )
            collector = SnapshotCollector(build_backend(cfg.backend))

    global _server_metrics_collector
    _server_metrics_collector = collector
    try:
        if collector is not None and run_cfg.procs > 1:
            asyncio.run(collector.start())
        started_at = datetime.now(timezone.utc)
        raw = run_shards(_shard_entry, vars(args), run_cfg.procs)
        ended_at = datetime.now(timezone.utc)
        if collector is not None and run_cfg.procs > 1:
            asyncio.run(collector.end())
    finally:
        _server_metrics_collector = None

    if collector is not None:
        window = "measured" if run_cfg.procs == 1 else "whole_run"
        server_metrics = finish(collector.result(), window=window)

    summary = aggregate(raw)

    meta = _run_metadata(
        args,
        dataset=cfg.dataset.name,
        backend=cfg.backend.name,
        build=build,
        started_at=started_at,
        ended_at=ended_at,
    )

    # The section's `raw` block is the full two-snapshot scrape: too big for
    # the inline copy, which exists to be read, and goes to its own file.
    display_metrics = None
    if server_metrics is not None:
        display_metrics = {k: v for k, v in server_metrics.items() if k != "raw"}
    payload: dict[str, Any] = {"meta": meta, "summary": summary}
    if display_metrics is not None:
        payload["server_metrics"] = display_metrics
    print(json.dumps(payload, indent=2))

    if args.output:
        out = args.output.rstrip("/")
        write_summary_json(
            summary, f"{out}/summary.json", meta=meta, server_metrics=display_metrics
        )
        write_summary_csv(summary, f"{out}/summary.csv")
        if server_metrics is not None:
            write_server_metrics(server_metrics, out)
        if args.raw:
            write_raw_ndjson(raw, f"{out}/raw.ndjson")
        print(f"reports written to {out}/")

    return 0


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
    parser = argparse.ArgumentParser(prog="ltm100", description="LTM100 load benchmark.")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, help="path to YAML config")
    common.add_argument("--users", type=int, default=10, help="virtual users")
    common.add_argument("--seed", type=int, default=0, help="RNG seed")

    run = sub.add_parser("run", parents=[common], help="run a benchmark")
    run.add_argument("--scenario", required=True, help="scenario name")
    g = run.add_mutually_exclusive_group()
    g.add_argument("--duration", type=float, default=0.0, help="run seconds (0=off)")
    g.add_argument("--ops", type=int, default=0, help="total ops cap (0=off)")
    run.add_argument("--global-concurrency", type=int, default=0, help="max in-flight")
    run.add_argument("--warmup", type=float, default=0.0, help="warmup seconds")
    run.add_argument("--preingest", action="store_true", help="pre-ingest memories before run")
    run.add_argument(
        "--preingest-fraction",
        type=float,
        default=1.0,
        help="fraction of each user's memories to pre-ingest",
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
        "--answer-time",
        type=float,
        default=0.0,
        help="chat-replay: mean seconds the LLM spends generating an answer "
        "after a user turn (Exponential; 0 = back-to-back, default). Applied "
        "uniformly to all users",
    )
    run.add_argument(
        "--user-gap",
        type=float,
        default=0.0,
        help="chat-replay: mean seconds the user takes before the next turn "
        "(Exponential; 0 = back-to-back, default). Applied uniformly to all users",
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
    run.add_argument("--output", default=None, help="output dir for reports")
    run.add_argument("--raw", action="store_true", help="also write raw.ndjson")
    run.add_argument("--no-delete-on-exit", action="store_true", help="keep user state")
    run.set_defaults(func=_run)

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
