# LTM100 — Multi-User Load Benchmark for Long-Term Memory Systems

> Status: **Implemented (baseline verified against a live MemMachine
> server).** This document is the design reference; the README is the fast
> entry point and `docs/` holds the per-topic detail.
> Last updated: 2026-09-23

## 1. Purpose

LTM100 is a benchmark for evaluating Long-Term Memory (LTM) software solutions
(e.g. MemMachine, Mem0) in a **server-client, multi-user** setting. Its core is
not retrieval/answer quality — it is **how an LTM server behaves when many users
repeatedly perform `add`, `search`, and mixed add/search workloads against a
single endpoint.**

The benchmark generates realistic and synthetic load, drives a running LTM
server through its client API, and extracts performance/load metrics
(latency, throughput, concurrency behavior). It is built to be **reusable and
publicly releasable**: datasets and LTM backends are pluggable, the run
configuration is simple, and defaults are sensible.

### What this benchmark measures

- Storage performance / throughput (`add`).
- Multi-user behavior under concurrency (`add`, `search`, mixed).
- Scalability / concurrency: QPS, latency percentiles (p50/p99), error rate.

### What this benchmark does NOT measure

- Retrieval / recall quality (no precision/recall/MRR/answer correctness).
- Memory isolation correctness as a dedicated test. Per-user scoping
  (`user_id` → backend tenant key) is assumed to be implemented correctly by
  each backend; isolation is an implicit property of correct per-user behavior,
  not a separate assertion.
- Server-side resource usage (CPU/mem/IO). Those are collected **by the
  server side separately**; LTM100 only collects client-observable metrics.

## 2. Goals & Non-Goals

**Goals**
- Pluggable datasets (LongMemEval first; BEAM, LoCoMo, others later).
- Pluggable LTM backends (MemMachine, Mem0, and Supermemory).
- Pluggable transport (REST first; MCP later) under a unified backend adapter.
- Multiple load scenarios: pure load tests and realistic per-user patterns.
- Reproducible (seeded), with variance runs available.
- Easy to adopt: minimal config, clear plugin contracts, good docs.

**Non-Goals**
- Answer-quality evaluation.
- Client-side process/container isolation per user (server already isolates).
- Built-in server resource monitoring.

## 3. Architecture Overview

Three pluggable axes, plus a load core and a metrics recorder:

```
                 +------------------------+
                 |   Scenario (op mix +   |
                 |   data; closed/open    |
                 |   consume schedule)    |
                 +-----------+------------+
                             |
                             v
   +-------------------+   +------------------+   +-------------------+
   |  DatasetAdapter   |-->|   Load Core      |-->|  MetricsRecorder  |
   |  (per-user add    |   | (asyncio users)  |   | (per-request,      |
   |   + turn streams) |   |                  |   |  by op type)       |
   +-------------------+   +--------+---------+   +-------------------+
                                    |
                                    v
                          +-------------------+
                          |  LTMClient (async)|  <-- backend adapter
                          +---------+---------+
                                    |
                          +---------+---------+
                          |  Transport (REST /|
                          |  MCP / ...)       |
                          +-------------------+
                                    |
                                    v
                          [ Running LTM Server ]
```

- **DatasetAdapter**: turns a raw dataset into per-user streams of `add`
  payloads (and optional dialogue `turn_stream` for chat-replay). Knows
  nothing about backends or search queries — search queries are
  content-derived by the scenarios.
- **LTMClient** (backend adapter): async `add` / `search` against a specific
  backend, with per-user tenant scoping. Knows nothing about datasets or
  scenarios.
- **Transport**: the wire protocol under a backend adapter (REST first, MCP
  later). Backend adapters delegate to a transport.
- **Load Core**: orchestrates N virtual users (asyncio tasks), drives them
  through a Scenario, and records metrics. By default a single process; can
  shard the users across several OS processes (`--procs`) so a fast server is
  not bottlenecked on one event-loop core.
- **Scenario**: defines *how* a user emits requests (inter-arrival, op mix,
  termination). Closed and open models share one runner.
- **MetricsRecorder**: collects per-request timing by op type; emits a
  summary + optional raw stream.

## 4. Pluggable Interfaces

All interface contracts below are the design intent; exact signatures are
finalized at implementation time.

### 4.1 DatasetAdapter

A dataset adapter exposes per-user workloads without leaking the raw dataset
shape into the core.

```python
class DatasetAdapter(Protocol):
    name: str

    def users(self, n_users: int, *, seed: int) -> list[UserId]:
        """Return n_users virtual-user identifiers (may replicate samples)."""

    def memory_stream(self, user: UserId) -> Iterator[MemoryItem]:
        """Yield add payloads for this user (in ingestion order)."""

    def turn_stream(self, user: UserId) -> Iterator[Turn]:
        """Optional: structured conversation turns, for chat-replay."""
```

- `MemoryItem`: `content` + optional `timestamp`, `producer`, `role`,
  `metadata`.
- `Turn`: `(role, items)` for dialogue datasets (optional; used by
  `chat-replay`).
- **No `query_stream`.** Search queries are **content-derived** by the
  scenarios — built from the user's own `memory_stream` items (for
  `search-load` / `mixed`) or from `turn_stream` user-turn content (for
  `chat-replay`). A dataset evaluation question is therefore not exposed as
  a search stream; this keeps the query pool large (one query per stored
  unit) so cycling does not naively repeat a single query and warm a server
  result cache.
- `QueryItem`: query string + optional expected fields (gold answer etc. are
  **not** scored — kept only for optional traceability/debugging).
- **Replication by default**: `n_users` is independent of dataset size; an
  adapter maps virtual users onto dataset samples (1:1 when enough samples,
  replication otherwise). The virtual-user count we drive is what matters,
  not the dataset's own user count.
- LongMemEval mapping: a sample's `haystack_sessions` → one user's
  `memory_stream` (flattened, chunked <=3000 chars) **and** `turn_stream`
  (per-turn role + chunked items, for chat-replay). Both streams preserve the
  source turn role on every `MemoryItem`; `chat-replay` also fills a missing
  item role from its enclosing `Turn.role`.
- Nebius mapping: seeded trajectory indices are distributed across the full
  virtual-user pool, replicating only when there are fewer trajectories than
  users. A finite `memory_stream` projects activity for generic workloads;
  the optional `AgentDataset.task_stream` in `ltm100/agent.py` retains independent
  `AgentTask` records. It does not imply dialogue support or an LTM call policy.
  See [Nebius](docs/nebius.md); coding-agent scheduling is a separate extension.

### 4.2 LTMClient (backend adapter)

```python
class LTMClient(Protocol):
    name: str

    async def setup(self, users: list[UserId]) -> None:
        """Per-run provisioning (e.g. create a project/tenant per user)."""

    async def add(self, user: UserId, items: list[MemoryItem]) -> list[str]:
        """Store memories for user; return backend ids."""

    async def search(self, user: UserId, query: QueryItem) -> list[ResultItem]:
        """Retrieve memories for user (scoped to this user only).

        The search depth (`top_k`) is carried on `query.top_k` (default 20,
        set via the scenario `top_k` param / `--top-k`), not as a separate
        argument. Server-side search knobs (`expand_context`, `filter`) are
        likewise carried on the `QueryItem` (default inert — the adapter omits
        them from the wire payload when unset); see §7.2."""

    async def teardown(self, users: list[UserId], *, delete: bool) -> None:
        """Optional cleanup (delete per-user state) per run."""
```

- Per-user scoping is the adapter's responsibility: it maps `UserId` to the
  backend's tenant key (MemMachine: `org_id`/`project_id` → `session_key`;
  Mem0: namespaced `user_id`; Supermemory: namespaced hashed `containerTag`).
- The adapter is async. Mem0 uses its self-hosted REST API, so the load path
  does not run a synchronous SDK inside the client process.
- `setup`/`teardown` are out-of-measurement phases.
- A backend may expose a `health()` probe so the run report records the
  server's own version (`meta.build`) — a throughput number is not
  reproducible without the build that produced it. Backends without it are
  simply left blank.
- An adapter that cannot honour a server-side search knob it was asked for
  (e.g. the MCP transport has neither `expand_context` nor `filter`) must
  **raise** rather than silently run a baseline search under the label of a
  filtered/expanded one — otherwise the error rate would lie.
  Supermemory explicitly disables unsupported context expansion with a warning,
  as documented in its [backend guide](docs/supermemory.md); no neighboring
  episodes are claimed. Existing adapters retain their current error behavior.

### 4.3 Transport (under a backend adapter)

```python
class Transport(Protocol):
    async def request(self, op: str, payload: dict) -> dict: ...
```

- REST is the first transport. MCP is a second transport with the **same**
  `LTMClient` contract; choosing it is a config switch, not a code fork. The
  MCP adapter imports lazily so a plain `pip install -e .` (without the
  `[mcp]` extra) can still print `--help`; the extra is required only when an
  MCP backend is actually selected.
- Backends that have only an SDK (no server) implement `LTMClient` directly.
- **Retries** live at the transport: only *connection-level* failures
  (timeout, connection error) are retried (with exponential backoff,
  `retries`, default 0). An HTTP error status is a real answer from the
  server and is **never** retried — retrying it would understate the error
  rate, which is a metric.

## 5. Load Model

Hybrid: both **closed** and **open** models are supported, sharing one runner.

### 5.1 Closed model (load test) — implemented first

- Fixed `N` virtual users, each with one or more session lanes looping:
  emit request → await response → (optional think time) → next request.
- Per-session **in-flight = 1**. Profiled `chat-replay` supports group-specific
  `concurrent_sessions`; lanes remain sequential and share the backend user.
- Optional **global concurrency cap** (`asyncio.Semaphore(K)`) to test "max
  concurrency K" load independent of N.

### 5.2 Open model (mixed) — implemented second

- Sessions arrive over time per an arrival process (Poisson by default, custom
  inter-arrival distributions pluggable).
- Each session performs a bounded number of ops then leaves.
- Concurrency is a *result* of arrival rate vs. service rate, not fixed.
- Profiled `chat-replay` may cap active sessions per user through
  `max_sessions_per_user`, required for every group under the open model;
  saturated-user arrivals are rejected without changing the configured
  arrival rate.
- **Congestion policy** (decided at implementation): when arrival rate
  exceeds server capacity, define a bounded queue / rejection, and record
  rejections + queue depth as metrics. No silent dropping.

### 5.3 Shared runner

Both models reduce to "a virtual user coroutine emits requests over time";
the difference is the *emit schedule* (closed: think-time loop; open:
inter-arrival). The runner is the same; the Scenario provides the schedule.

### 5.4 Multi-process load generation (`--procs`)

One asyncio event loop saturates a single CPU core well before a healthy
server does, so past a few dozen users a single-process run reports the
**generator's** ceiling rather than the target's. `--procs N` runs the same
`LoadRunner` in N OS processes, each driving a disjoint slice of the virtual
users, and pools their raw `OpResult`s.

- Shards are **spawned** (not forked): a forked child inherits the parent's
  event loop and open sockets, which asyncio does not support.
- Users are partitioned round-robin (`runner.shard_users`), so an ordered
  dataset does not hand one shard all the large conversations.
- Whole-run budgets are divided across shards so an undivided value is not
  applied N times over: `--ops`, `--global-concurrency`, `--queue-bound`,
  and `--arrival-rate` are split (the rate is divided, not the count);
  `--session-ops` is per-session and `--users` is partitioned by `shard_users`,
  so neither is divided.
- Raw results are pooled (not per-shard summaries) so percentiles are
  computed over the whole population by the same `aggregate` used for a
  single-process run — no approximation from per-shard percentiles.
- `--procs 1` is the original single-process topology and is exactly the
  same code path.

See [`docs/load-models.md`](./docs/load-models.md) for the implemented
mechanics (closed loop, Poisson arrivals, the bounded-queue rejection
policy, multi-process sharding) and when to use each model.

## 6. Scenarios

A Scenario decides, per virtual user, the **op mix** and **emit schedule**.

```python
class Scenario(Protocol):
    name: str

    def plan(
        self, user: UserId, dataset: DatasetAdapter, rng_state: dict
    ) -> Iterator[Op]:
        """Yield the sequence of ops for this user, in order. `rng_state`
        carries the seeded RNG state so schedules are reproducible."""

    def validate(self, dataset: DatasetAdapter) -> None: ...
```

Initial scenarios (`chat-replay` is the primary workload; the rest are
auxiliary load probes):
1. **`chat-replay`**: replay a chatbot-with-LTM workload over the dataset's
   `turn_stream` (recall before a user turn, then ingest the turn), with a
   configurable recall cadence (`search_every`) and LLM answer / user think
   time (`answer_time`, `user_gap`). The primary workload.
2. **`add-load`**: each user streams `memory_stream` back-to-back (wrapping),
   max concurrency. Pure storage throughput.
3. **`search-load`**: users run pre-ingested, content-derived searches
   forever. Pure search throughput/latency.
4. **`mixed`**: a controllable add/search mixture (op mix via
   `search_weight`), needs no `turn_stream` so it works with the synthetic
   dataset. Under the open model its per-session slice gives a quick
   congestion probe; under closed it loops like any other scenario.

Every scenario runs under **both** closed and open load models. The op mix
(add vs search) is owned by the Scenario plan for **both** models — the open
model's arriving sessions consume a bounded number of ops from the same
`plan()` interface the closed model loops over. There is no runner-level
op-mix weight; `mixed` takes a `search_weight` constructor param instead.
All scenario plans are **infinite** (they wrap their stream), so a duration
run sustains load instead of going idle when a finite stream is exhausted.

## 7. Metrics

All metrics are **client-observable** and **separated by op type**
(`add` vs `search`).

Per-request recorded fields: `op_type`, `user_id`, `started_at`, `ended_at`,
`latency_ms`, `status` (ok / error / rejected), `error_kind`, `n_items` (items
stored on add / results returned on search).

Aggregated summary:
- Total offered count across op types and wall-clock seconds.
- Overall and per-op offered, accepted, successful, error, and rejected counts
  and rates. `throughput_ops_s` / `qps` mean successful throughput; explicit
  `*_ops_s` fields preserve every population needed for overload analysis.
- Per op type: successful-request latency (mean, p50, p90, p95, p99, max),
  rejection rate (rejected / offered), error rate (backend errors / accepted),
  error-kind breakdown, and **items** (`mean` results per successful op, `empty`
  count, `empty_rate`). Zero-time rejections never enter service percentiles.
  `empty_rate` matters for search: a run where every query returns zero results
  still has a 0% error rate, so it distinguishes a working search from a silent
  one.
- Concurrency (observed concurrent in-flight over time, for open model).

### 7.2 Report metadata (`meta`)

The summary JSON carries whole-run metadata: `dataset`, `backend`, `scenario`,
the load and scenario parameters, `procs`, `started_at`, `ended_at`, and a
**server build** probe. Values such as `ops`, `global_concurrency`,
`arrival_rate`, and `queue_bound` are the requested totals, not one process
shard's share. `started_at` and `ended_at` bracket worker execution, including
setup, optional pre-ingest, measurement, and teardown. Before the run, the harness
asks the backend's `health()` for the server's version (`meta.build`) — the
one thing the harness cannot infer, and the cause of silent mismatches when
two runs from two server builds are compared. A failed probe does not cost
the run (the field reports `<unavailable: ...>`); a backend without
`health()` is simply left blank.

Recording:
- **Default**: in-memory per-request list, aggregated post-run. Good for
  small/medium scale and full reproducibility/debugging.
- **Optional**: NDJSON streaming (large scale), enabled via flag.
- Percentiles computed post-run by the nearest-rank method (no numpy).

Server-side **resource** metrics (CPU/mem/IO) are not collected here. Server-side
**latency** metrics, however, can be pulled into the report opt-in — see §7.3.

### 7.3 Server-side latency metrics (`--server-metrics`)

A benchmark's client latency is the server's latency plus queueing; to attribute
a slow search to a server phase (embedding call vs. vector query vs. reranking)
the server has to say so itself. A backend that exposes its own Prometheus
metrics endpoint declares it with the class attribute
`supports_server_metrics`, and the CLI scrapes it — two snapshots bracketing
the run, and every histogram reported as the difference. With
`--server-metrics-interval`, intermediate snapshots additionally expose the
same breakdown as a time series.

- **Declared, not guessed.** `--server-metrics` on a backend without the
  attribute warns and runs without the section; on a backend that declares it
  but whose endpoint fails (older build), the pre-run probe disables it the
  same way. Neither case costs the run anything.
- **Deltas, because Prometheus counters are cumulative.** A negative delta is
  a counter reset (container restart mid-run): the series is excluded with a
  warning rather than averaged into a wrong number. `Δcount == 0` reads
  "not executed"; a quantile landing in the `+Inf` bucket reads "beyond
  buckets" rather than being extrapolated.
- **The window is the measured window** (pre-ingest, warm-up, and teardown
  excluded) via runner measure-hooks. With `--procs > 1`, every worker waits
  at start/end measurement boundaries while the parent takes the snapshots,
  so the server and pooled client results cover the same interval.
- **What is reported**: MemMachine's five add-pipeline phases
  (segmentation, derivation, embedding, segment_store, vector_store), four
  search phases (embedding, vector_query, segment_query, scoring), and the two
  request-level `http_request_duration_seconds` paths — mean from `Δsum/Δcount`,
  p50/p90/p99 interpolated inside the server's own bucket edges (resolution is
  therefore the server's, and phases faster than the smallest bucket carry no
  distribution). The full scrape, including the secondary component metrics
  (embedder, vector store, segment store, ...) used for cause attribution,
  goes to `server_metrics_raw.json`; the table goes to
  `summary.json["server_metrics"]` and `server_metrics.csv`.
- **Periodic breakdowns are adjacent deltas.** The sampler uses fixed
  monotonic deadlines on a dedicated thread, preserving the load generator's
  event loop. `server_metrics_timeseries.csv` diffs each snapshot only against
  its immediate predecessor and keeps the final partial interval. A failed
  scrape invalidates intervals touching that boundary; it is never bridged to
  a later snapshot and mislabeled as one interval. Slow scrapes skip missed
  deadlines instead of sending catch-up requests.
- **The row table belongs to the current backend.** The phase/http series
  listed above are MemMachine's; the parser, the delta arithmetic, and the
  report writers behind it know nothing of that, and a second backend that
  implements the metrics query contributes its own row table in place of
  this one.
- **Observation never breaks observation**: a boundary snapshot failure yields
  `status: "failed"` for the aggregate; a periodic failure yields a partial
  time series with the affected intervals marked failed. Both still produce a
  normal benchmark report.

## 8. Run Lifecycle

1. **Load config** (YAML) — backend endpoint/auth, transport, adapter choices.
2. **Resolve adapters** — dataset + LTM client (+ transport).
3. **Provision** (`LTMClient.setup`) — per-user tenants created. (out of measure)
4. **Optional pre-ingest** — fill each user's memories before the workload;
   excluded from metrics. `--preingest-fraction` keeps the legacy fractional
   mode, while `--preingest-items-per-user` incrementally consumes an exact
   per-user prefix for large-corpus experiments. The modes are mutually
   exclusive and run under the global concurrency cap. A zero size is a no-op,
   a short stream or ingestion failure aborts before warm-up or measurement,
   and the report records successfully submitted input-item counts.
5. **Optional warm-up** — drive the selected workload for `--warmup` seconds
   against the real backend. Requests can update backend state but are not
   recorded and do not consume the measured duration or operation budget.
   In-flight warm-up work drains before measurement starts.
6. **Measured run** — scenario drives users; MetricsRecorder collects.
7. **Drain** — in-flight requests complete (or timeout).
8. **Teardown** (`LTMClient.teardown`, `delete=True`) — optional per run;
   also exposed as a standalone cleanup command.
9. **Aggregate & report** — summary JSON/CSV + optional fixed-interval E2E
   time series + optional raw NDJSON + optional server-metrics section (§7.3),
   whose aggregate and periodic snapshots are taken inside step 6's
   boundaries when `--server-metrics` is on.

Termination: count-based (total K ops) **or** time-based (T seconds). Ramp-up
is optional; warm-up time is excluded from steady-state metrics.

## 9. Configuration

**YAML** (stable, per-environment): backend endpoint/auth, transport choice,
dataset adapter, LTM client adapter, defaults.

**CLI** (per-run, changed often): `--users N`, `--scenario`, `--duration` /
`--ops`, `--seed`, `--model`, `--global-concurrency`, `--warmup`, `--rampup`,
`--preingest`, `--preingest-fraction`, `--preingest-items-per-user`,
`--arrival-rate`, `--session-ops`, `--queue-bound`, `--search-weight`,
`--top-k`, `--query-limit`, `--think`, `--search-every`, `--answer-time`,
`--user-gap`, `--expand`, `--filter`, `--procs`, `--raw`,
`--time-series-interval`, `--no-delete-on-exit`, `--output`. The MCP import is
lazy, so `--help` works
without the `[mcp]` extra. See `ltm100 run --help` for the authoritative list.

Connection retries and the streaming JSON loader are backend/dataset
options (set in the YAML, not on the CLI): the MemMachine backend takes a
`retries` option (default 0; connection-level failures only); the
LongMemEval adapter reads large local files with `ijson` (a runtime
dependency) so only `dataset.length` samples are materialized instead of the
whole document.

## 10. Reproducibility

- Seeded RNG by default (`--seed`). Used for: virtual-user→sample mapping,
  op ordering, think time, inter-arrival, op mix.
- Different seeds produce variance runs; same seed + same config reproduces.
- Wall-clock timing is inherently non-deterministic (network/server); the
  *load shape* (order, mix, arrival) is deterministic under a seed.
- With `--procs N`, the per-user seed is derived from the run seed and the
  user id (not the shard index), so the same run shape reproduces regardless
  of how many processes shard it.

## 11. Project Layout (proposed)

```
ltm100/
  ltm100/
    core/            # load core, runner, scenarios, multiproc
    adapters/
      datasets/      # longmemeval.py (ijson streaming), synthetic.py, ...
      backends/       # memmachine.py, memmachine_mcp.py, mem0.py, ...
      transports/     # rest.py (retry loop), mcp.py
    metrics/         # recorder, aggregation, report
    config.py        # lazy MCP import
    cli.py           # sharding, health/build probe, report
  datasets/          # adapter-specific data access (not raw data)
  examples/          # sample configs + run commands
  docs/
  tests/
  README.md
  DESIGN.md          # this file
```

## 12. Initial Baseline (implemented)

First concrete adapters, end-to-end, **implemented and verified against a live
MemMachine server** via a smoke run:
- Dataset: **LongMemEval** (`xiaowu0162/longmemeval-cleaned`, `longmemeval_s_cleaned`;
  https://github.com/xiaowu0162/longmemeval, MIT), loaded at runtime from
  HuggingFace **or** a pre-downloaded local JSON file (`path` option, streamed
  with `ijson` so only `length` samples are materialized instead of the whole
  multi-GB document; the data is not redistributed with LTM100 — see the
  adapter docstring for the citation). A **Synthetic** adapter (`synthetic`) is
  also provided for fast, dependency-free load testing, with an optional
  `categories` option that writes
  `metadata.category` for `--filter` to select on.
- Backend: **MemMachine** over **REST** (`/api/v2`), with
  `UserId → {org_id, project_id}` → `session_key = f"{org_id}/{project_id}"`,
  and a second transport over **MCP** (same `LTMClient` contract). The REST
  transport retries connection-level failures only (`retries` option, default
  0). Both adapters support the server-side search knobs `expand_context` and
  `filter` over REST; the MCP adapter **refuses** them (its `search_memory`
  tool exposes neither) rather than silently ignoring them.
- Scenarios: `chat-replay` (primary), `add-load`, `search-load`, `mixed` —
  all run under both closed and open models; all plans wrap their streams.
  (`add-search-mixed` was retired early on — its `search_every` cadence
  moved to `chat-replay`.) Scenario params: `--think`, `--search-every`,
  `--search-weight`, `--top-k`, `--answer-time`, `--user-gap`, `--expand`,
  `--filter`.
- CLI: `ltm100 run`, `ltm100 cleanup`; `--procs N` shards the run across
  processes; reports: `summary.json` (incl. `meta.build` from the server
  health probe), `summary.csv`, optional `raw.ndjson`.

Verified: health, add/search, per-user isolation, count- and time-based
termination, global concurrency cap, reproducibility (same seed), report
generation (incl. overall QPS, per-op `items.empty_rate`, `meta.build`),
multi-process sharding (`--procs`), streaming dataset load, retries, and
cleanup (teardown delete). No code bugs found.

### Backend memory-type selection

MemMachine REST defaults both operations to `types: ["episodic"]`, preserving
existing YAML and wire behavior. Optional `add_memory_types` and
`search_memory_types` independently select episodic, semantic, or both.
Selection belongs to the backend adapter; scenarios, datasets, and runner
budgets are unchanged. Semantic selection requires a read-only enabled-state
preflight before workload setup; other adapters reject these options.
Semantic responses are normalized without invented scores or combined top-k
truncation. Input acknowledgements do not imply semantic extraction completion.
See [memory types](docs/memory-types.md) for the full contract and restrictions.

### Known caveat: MemMachine `projects/list` is eventually consistent

After `teardown(delete=True)` succeeds (server returns 204 and logs
`Deleted session`), an immediate `projects/list` call may still show the
project. It disappears shortly after. Cleanup verification must wait a moment
or check delete response status, not trust an immediate list.

## 13. Status of Open Questions

Resolved during implementation:
- Async signatures: `Protocol` for `LTMClient` / `DatasetAdapter` / `Transport`;
  concrete classes (`MemMachineClient`, `LongMemEvalAdapter`, `RestTransport`).
- Per-session in-flight = 1 is enforced in the runner. Profiled chat users may
  own several closed-model lanes or dynamically admitted open-model lanes.
- NDJSON raw format: per-request `{op_type, user_id, started_at, ended_at,
  latency_ms, status, error_kind, n_items}`.
- **Warm-up pre-ingest** (§8 step 4): the runner pre-ingests each user's
  memories (fraction configurable) before the measured run, under the global
  concurrency cap. Excluded from metrics.
- **Open model + congestion policy**: a Poisson arrival process spawns
  arriving sessions, each consuming a bounded number of ops from the Scenario
  plan (`mixed`). A bounded queue on the global concurrency cap rejects
  overload as `status="rejected"` (zero latency, `error_kind="queue_full"`).
  Op mix is owned by the Scenario plan (not a runner-level weight), so the
  open and closed models share one Scenario interface.
- **MCP transport**: a second transport under the same `LTMClient` contract
  (`MemMachineMcpClient` + `McpTransport` over `fastmcp`). The measured
  add/search ops call MemMachine's `add_memory`/`search_memory` MCP tools;
  lifecycle (project create/delete) stays on REST, since MCP has no
  project-management tools (hybrid lifecycle). Tenancy is passed as MCP tool
  arguments. The MCP `add_memory` writes all memory types (episodic + semantic),
  unlike the default episodic-only REST add, so MCP add latency is not directly
  comparable to REST add latency; documented rather than worked around. The
  MCP adapter imports lazily (`config._mcp_backend`) so a plain install
  without the `[mcp]` extra can still print `--help`; the extra is required
  only when an MCP backend is selected. The MCP `add_memory` tool has no
  metadata or role field, so the adapter **refuses** items carrying metadata
  (a metadata `--filter` would otherwise select nothing for invisible
  reasons) and intentionally omits role so dialogue workloads remain
  runnable; `search_memory` exposes neither `expand_context` nor
  `filter`, so the adapter **refuses** those knobs rather than silently
  running a baseline search under their label.
- **chat-replay LLM timing**: `chat-replay` models the LLM answer time
  (`answer_time`) and the user's think/typing time (`user_gap`) as
  Exponential-mean delays attached before the following assistant add and
  the next user op, respectively, defaulting to 0 (back-to-back). Search depth
  is configurable via `top_k` (default 20). A versioned chat profile assigns
  weighted user groups with overrides for timing, recall cadence/depth,
  closed-model `concurrent_sessions`, and open-model
  `max_sessions_per_user`. Source conversation boundaries remain intact;
  request and session admission metrics remain separate.
- **mixed think timing**: both ADD and SEARCH operations receive
  `uniform(0, think)` delay. ADD delay uses a separate seeded RNG stream so
  enabling it does not change the established add/search operation mix.

Still open / next work is tracked in GitHub issues rather than duplicated in
this design document.
4. **Additional datasets** (BEAM, LoCoMo) via the `DatasetAdapter` extension
   (must implement `memory_stream`, and `turn_stream` if dialogue).

## 14. Glossary

- **Virtual user**: an asyncio task simulating one user, identified by a
  `UserId`; maps to a backend tenant.
- **Tenant key**: backend-specific per-user isolation key (MemMachine:
  `session_key`).
- **In-flight**: a request sent but not yet responded to.
- **Closed model**: fixed number of concurrent users, each looping.
- **Open model**: users arrive over time; concurrency is emergent.
- **Op**: a single `add` or `search` request.
