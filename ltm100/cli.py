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
from ltm100.metrics.report import write_raw_ndjson, write_summary_csv, write_summary_json

logger = logging.getLogger(__name__)


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

    runner = LoadRunner(
        client=backend, dataset=dataset, scenario=scenario, config=run_cfg
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
    except Exception as e:  # a failed probe must not cost the run
        return {"build": f"<unavailable: {type(e).__name__}>"}
    if not reported:
        return {}
    return {
        "build": reported.get("version") or "<unreported>",
        "service": reported.get("service") or "<unreported>",
    }


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

    raw = run_shards(_shard_entry, vars(args), run_cfg.procs)
    summary = aggregate(raw)

    meta = {
        "dataset": cfg.dataset.name,
        "backend": cfg.backend.name,
        **build,
        "scenario": args.scenario,
        "users": run_cfg.users,
        "seed": run_cfg.seed,
        "duration": run_cfg.duration,
        "ops": run_cfg.ops,
        "global_concurrency": run_cfg.global_concurrency,
        "procs": run_cfg.procs,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    print(json.dumps({"meta": meta, "summary": summary}, indent=2))

    if args.output:
        out = args.output.rstrip("/")
        write_summary_json(summary, f"{out}/summary.json", meta=meta)
        write_summary_csv(summary, f"{out}/summary.csv")
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
    if not getattr(args, "duration", 0) and not getattr(args, "ops", 0):
        if args.command == "run":
            parser.error("run requires either --duration or --ops")
    # The run path owns its own event loops (one per shard process), so only
    # the coroutine commands are wrapped here.
    result = args.func(args)
    if inspect.iscoroutine(result):
        return asyncio.run(result)
    return result


if __name__ == "__main__":
    sys.exit(main())
