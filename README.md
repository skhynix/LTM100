# LTM100

LTM100 is an end-to-end multi-user **load benchmark** for Long-Term Memory (LTM) systems
(e.g. MemMachine, Mem0). It drives many virtual users performing `add`,
`search`, and `add&search` operations against a single LTM endpoint and
reports client-observable performance metrics: throughput, QPS, latency
percentiles, and error rate, all split by operation type.

This benchmark measures **load, concurrency, and scalability behavior** —
not retrieval or answer quality. There are no precision/recall/MRR metrics.
Server-side resource metrics (CPU, memory, etc.) are collected separately by
the server itself.

## Status

Version: **v0.4.1** (see [Versioning](#versioning)).

Early development. Datasets and LTM backends are pluggable; the initial
baseline is the LongMemEval dataset + the MemMachine backend over REST.

Implemented:
- Closed and open load models (every scenario runs under both).
- Scenarios: `chat-replay` (the primary workload), `add-load`, `search-load`,
  `mixed`. All wrap their data streams to sustain load and search on
  content-derived queries.
- Congestion policy (bounded queue with rejection) for the open model.
- Multi-process load generation (`--procs N`), so a fast server is not
  bottlenecked on one event-loop core.
- Warm-up / pre-ingest before the measured run.
- Datasets: LongMemEval (local file or HuggingFace; large local files are
  streamed with `ijson`), synthetic (with optional `categories` for
  `--filter`).
- Backends: MemMachine (REST and MCP transports) and Mem0 OSS (REST); REST
  retries connection-level failures only (`retries` option).
- Server-side search knobs: `--expand` (expand_context) and `--filter`
  (metadata filter), forwarded to every search; the MCP backend refuses
  them loudly rather than ignoring them.
- Configurable scenario parameters on the CLI (`--think`, `--search-every`,
  `--search-weight`, `--top-k`, `--answer-time`, `--user-gap`, `--expand`,
  `--filter`).
- Reports: summary JSON/CSV + optional raw NDJSON, with per-op
  `items.empty_rate` and the server build recorded in `meta.build`.

Planned (see [DESIGN.md](./DESIGN.md) for the full roadmap):
- Per-user in-flight > 1; per-user-group finer control;
  configurable memory types; additional datasets (BEAM, LoCoMo).

## Install

```sh
pip install -e ".[dev]"
# To use the LongMemEval adapter via HuggingFace also:
pip install -e ".[datasets]"
# To use the MemMachine MCP transport also:
pip install -e ".[mcp]"
```

The `ltm100` CLI is the entry point. In this environment it is invoked as
`python -m ltm100.cli` if the console script is not on PATH.

## Configuration

A run takes two inputs:

- A **YAML config file** (stable, per environment): backend endpoint/auth,
  transport, and the dataset and backend adapter choices. See
  `examples/memmachine.yaml` and `examples/synthetic.yaml`.
- **CLI flags** (per run): number of users, scenario, duration/ops, seed,
  load model, concurrency, warm-up, and output. See `ltm100 run --help`.

Edit `examples/*.yaml` to point at your LTM server (`backend.base_url`) and
pick a dataset.

## Quick start

First, make sure your LTM server is up (e.g. MemMachine at
`http://localhost:8080`), then run one of the scenarios below. `chat-replay`
is the primary workload; the others are auxiliary load probes.

### Chatbot-LTM integration (`chat-replay`, the primary workload)

Replay a multi-turn dialogue as a chatbot-with-LTM would: before each user
turn, recall (search) against the user's utterance, then ingest both the
user and assistant turns. Requires a dataset with a `turn_stream`
(LongMemEval, not synthetic); the run fails loudly otherwise. The dialogue
wraps, so a `--duration` run replays it as many times as needed.

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 60 --seed 0 \
    --output out/chat-replay
```

Model the LLM answer time and the user's typing time so the load shape
resembles a real chatbot session (both default to 0 = back-to-back):

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 60 --seed 0 \
    --answer-time 2.0 --user-gap 3.0 \
    --output out/chat-replay
```

Run the same chatbot workload under realistic arrival timing (open model):
users arrive per a Poisson process and the congestion policy applies — the
recall cadence and turn content are unchanged.

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 30 --seed 0 \
    --model open --arrival-rate 2.0 --session-ops 12 \
    --global-concurrency 8 --queue-bound 4 \
    --output out/chat-replay-open
```

### Storage throughput (`add-load`, closed)

Pure add pressure — measure how fast the server ingests memories.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario add-load --users 20 --duration 30 --seed 0 \
    --output out/add-load
```

### Search throughput & latency (`search-load`, closed)

Pre-ingest memories, then loop searches. `--preingest` fills each user's
memories before the measured run; `--preingest-fraction` controls how much.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario search-load --users 20 --duration 30 --seed 0 \
    --preingest --preingest-fraction 1.0 \
    --output out/search-load
```

### Arrival-driven congestion probe (`mixed`, open)

A flat add/search mixture with a tunable search ratio — needs no dialogue,
so it runs against the synthetic dataset. Overload beyond the queue bound
is rejected; sweep `--arrival-rate` to find the server's capacity.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario mixed --users 8 --duration 20 --seed 0 \
    --model open --arrival-rate 5.0 --session-ops 6 \
    --queue-bound 4 --global-concurrency 4 --search-weight 0.8 \
    --preingest --preingest-fraction 0.5 \
    --output out/mixed
```

### MCP transport (same workload, MCP tools)

Drive add/search through MemMachine's `add_memory` / `search_memory` MCP
tools instead of REST — same `LTMClient` contract, so the workload and flags
are identical; only the config changes. Useful to compare REST vs MCP
overhead on the same load. (Requires `pip install -e ".[mcp]"`.)

```sh
ltm100 run --config examples/memmachine-mcp.yaml \
    --scenario chat-replay --users 20 --duration 30 --seed 0 \
    --output out/mcp-chat
```

### Server-side search knobs (`--expand`, `--filter`; REST only)

Forward `expand_context` and a metadata filter to every search so a run can
match another harness's search behaviour. The filter needs the corpus to
carry the field — the synthetic dataset writes `metadata.category` when its
`categories` option is set (uncomment it in the config):

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario search-load --users 20 --duration 30 --seed 0 \
    --preingest --expand 2 --filter metadata.category=cat_3 \
    --output out/filtered-search
```

### Scaling the client (`--procs`)

One asyncio process saturates a single core; a fast server can leave the
client as the bottleneck. Shard the users across N OS processes — the whole
run's budgets are divided across the shards:

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 100 --duration 60 --seed 0 \
    --procs 4 --output out/scaled
```

### Common flags

- `--duration SECONDS` or `--ops N`: how a run terminates (one is required).
- `--users N`: number of virtual users.
- `--seed N`: reproducible load shape (varies with N for variance runs).
- `--model closed|open`: load model (see [Load models](docs/load-models.md)).
- `--global-concurrency N`: cap total in-flight ops (0 = no cap).
- `--arrival-rate F` / `--session-ops N` / `--queue-bound N`: open-model knobs.
- `--procs N`: shard virtual users across N OS processes (default 1). One
  asyncio process saturates a single core before a fast server does; raise
  this when the client, not the server, is the bottleneck. Whole-run budgets
  (ops, global-concurrency, arrival-rate, queue-bound) are divided across
  shards; `--procs 1` is the original single-process topology.
- `--expand N`: (search-load, mixed, chat-replay) server-side `expand_context`
  — neighbouring episodes returned around each hit (default 0 = off, omitted
  from the request). REST only.
- `--filter EXPR`: (search-load, mixed, chat-replay) server-side metadata
  filter, e.g. `metadata.category=cat_3`. Needs the corpus to carry that
  field (synthetic's `categories` option writes it). REST only.
- `--rampup SECONDS`: stagger user start to avoid a thundering herd.
- `--search-weight F`: (mixed) fraction of ops that are search (0..1).
- `--top-k N`: (search-load, mixed, chat-replay) memories returned per search
  (default 20). Applied to all users.
- `--think SECONDS`: (mixed, chat-replay) max think-time jitter per op.
- `--search-every N`: (chat-replay) recall every N user turns (default 1).
- `--answer-time SECONDS`: (chat-replay) mean LLM answer time after a user
  turn (Exponential; 0 = back-to-back, default). All users.
- `--user-gap SECONDS`: (chat-replay) mean user think/typing time before the
  next turn (Exponential; 0 = back-to-back, default). All users.
- `--preingest` / `--preingest-fraction F`: pre-fill memories before the run.
- `--raw`: also write per-request `raw.ndjson`.
- `--no-delete-on-exit`: keep per-user state after the run.

See `ltm100 run --help` for the complete list.

## Backend setup

LTM100 talks to an LTM server through a pluggable **backend adapter**
(`LTMClient` contract). The supported backends are **MemMachine** and
**Mem0 OSS**:

- **MemMachine (REST)** — `examples/memmachine.yaml`. Points `backend.base_url`
  at the server (e.g. `http://localhost:8080`); `org_prefix` namespaces
  per-user projects (`session_key = f"{org_prefix}/user_{UserId}"`); one
  user = one MemMachine project, created on setup and deleted on teardown.
- **MemMachine (MCP)** — `examples/memmachine-mcp.yaml`. Same workload via
  `add_memory` / `search_memory` MCP tools. Lifecycle stays on REST (MCP has
  no project-management tools). Note `add_memory` writes all memory types
  (episodic + semantic), unlike the episodic-only REST add, so MCP add
  latency is not directly comparable to REST add latency. The MCP tools
  expose neither `expand_context`/`filter` nor item metadata, so the MCP
  backend **refuses** `--expand`/`--filter` and metadata-bearing items
  loudly — use the REST backend for those arms.
- **Mem0 OSS (REST)** — `examples/mem0.yaml`. Maps each virtual user to a
  namespaced Mem0 `user_id`; uses `POST /memories`, `POST /search`, and
  `DELETE /memories`. The default `infer: false` stores each input as one
  memory without LLM fact extraction, matching LTM100's item accounting and
  the MemMachine episodic-only baseline. Set `infer: true` explicitly to
  benchmark Mem0's extraction pipeline. Mem0 supports `--filter` but not
  `--expand`.

Verify the server is up before a run (MemMachine: `GET /api/v2/health`).
Further design details and how to add a new backend are described in
[`DESIGN.md`](./DESIGN.md).

## Datasets

The baseline dialogue dataset is [LongMemEval](https://github.com/xiaowu0162/longmemeval).
A **Synthetic** dataset is bundled for load testing.

## Reports

With `--output DIR`, LTM100 writes:

- `summary.json` — aggregated metrics with offered, accepted, successful,
  error, and rejected counts/rates. `throughput_ops_s` and `qps` are successful
  throughput; latency percentiles contain successful requests only. Rejection
  rate is rejected/offered and error rate is backend errors/accepted.
  `items.empty_rate` remains the fraction of successful searches returning
  nothing. `meta` records the run config plus the server's own build
  (`meta.build`, probed from `/api/v2/health`).
- `summary.csv` — the same summary as a flat table, with an overall `all` row
  (throughput/qps only; latency cells blank since mixing add/search latencies is
  ambiguous).
- `raw.ndjson` (with `--raw`) — one line per request.

## Cleanup per-user state

Per-run state (e.g. MemMachine projects) is deleted on exit by default. To
delete it without running a benchmark:

```sh
ltm100 cleanup --config examples/memmachine.yaml --users 50
```

> Note: MemMachine's `projects/list` is eventually consistent — an immediate
> list after delete may still show the project before it disappears. The
> delete itself is confirmed by the server's response.

## Documentation

This README is a fast entry point. Detail lives under [`docs/`](./docs/) and
[`DESIGN.md`](./DESIGN.md):

- [`docs/scenarios.md`](./docs/scenarios.md) — per-scenario data flow: how a
  virtual user behaves, concurrency, the `add`/`search` ops at the
  backend-contract level, plus the scenario comparison table.
- [`docs/load-models.md`](./docs/load-models.md) — the closed/open load
  models, the congestion/rejection policy, and when to use which.
- [`DESIGN.md`](./DESIGN.md) — full design: pluggable adapter contracts,
  metrics, run lifecycle, reproducibility, project layout, roadmap.

## Tests

```sh
pytest -q
```

## Versioning

Releases are marked with git tags (`vMAJOR.MINOR.PATCH`). The current release
is **v0.4.1**. Tag a release at a stable, documented milestone:

```sh
git tag v0.4.1
git push origin v0.4.1
```

During 0.x, each minor bump marks a meaningful, tested milestone (a coherent
set of features verified against a live server). Breaking changes bump the
minor version while still in 0.x.
