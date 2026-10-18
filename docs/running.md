# Running LTM100

This guide covers configuration, common workload commands, load controls,
backend-specific behavior, reports, and cleanup. For scenario internals, see
[`scenarios.md`](./scenarios.md); for closed and open scheduling, see
[`load-models.md`](./load-models.md).

## Configuration

A run combines two inputs:

- A **YAML config file** for the dataset, backend adapter, endpoint, and stable
  environment-specific options.
- **CLI flags** for the scenario, users, termination, load model, concurrency,
  pre-ingest, ramp-up, and output.

Start from [`examples/`](../examples/README.md), update `backend.base_url`, and
verify that the target LTM service is running. Scenario selection stays on the
CLI so the same backend/dataset config can drive multiple workloads.

LongMemEval supports all four scenarios. Synthetic data supports `add-load`,
`search-load`, and `mixed`, but not `chat-replay`, because it has no structured
dialogue `turn_stream`.

Nebius agent trajectories support the same generic add/search workloads via
`examples/nebius.yaml`. They preserve structured tasks for a future
coding-agent scenario and intentionally reject `chat-replay`. See the
[Nebius guide](nebius.md) for loading, user mapping, and corpus size controls.

## Workload examples

### Chatbot-LTM integration

`chat-replay` recalls against each user utterance and then stores the user and
assistant turns. It requires a dialogue dataset such as LongMemEval.

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 60 --seed 0 \
    --output out/chat-replay
```

Model the LLM answer gap and the user's reading/typing gap:

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 60 --seed 0 \
    --answer-time 2.0 --answer-time-variation 0.3 \
    --user-gap 3.0 --user-gap-variation 0.5 \
    --output out/chat-replay
```

Apply group-specific recall depth, timing, and closed-model session counts:

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --chat-profile examples/chat-profile.yaml \
    --users 100 --duration 60 --seed 0 \
    --global-concurrency 100 --output out/chat-replay-profiled
```

Run the same workload with arrival-driven sessions:

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 30 --seed 0 \
    --model open --arrival-rate 2.0 --session-ops 12 \
    --global-concurrency 8 --queue-bound 4 \
    --output out/chat-replay-open
```

Adding the same profile gives each group a distinct
`max_sessions_per_user` admission cap without changing `--arrival-rate`:

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --chat-profile examples/chat-profile.yaml \
    --users 100 --duration 30 --seed 0 \
    --model open --arrival-rate 20 --session-ops 12 \
    --global-concurrency 40 --queue-bound 20 \
    --output out/chat-replay-open-profiled
```

### Chat workload profiles

`--chat-profile PATH` is valid only with `chat-replay`. The versioned YAML
assigns whole users to weighted groups after a seeded shuffle. Largest-
remainder allocation makes group counts sum exactly to `--users`; assignments
are resolved before process sharding, so `--procs` does not change them.

Group fields override `defaults`, which override the corresponding CLI values:

- `think`, `search_every`, `answer_time`, `answer_time_variation`, `user_gap`,
  `user_gap_variation`, and `top_k` shape each group's conversation operations.
- `concurrent_sessions` is closed-model only. It creates that many fixed,
  sequential conversation lanes for each user.
- `max_sessions_per_user` is open-model only. It caps active Poisson-arriving
  sessions for each user and is required for every group in a profiled open
  run. This keeps every arrival on a boundary-preserving conversation lane.

All lanes for one user share the same backend tenant and memories. LongMemEval
`haystack_sessions` are distributed without overlap across the user's lanes;
the run fails before backend setup when a configured lane count exceeds the
available source conversations. `--global-concurrency` remains the final cap
on simultaneous backend requests under either model.

### Add throughput

`add-load` continuously ingests memory items. It works with both LongMemEval
and synthetic data.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario add-load --users 20 --duration 30 --seed 0 \
    --output out/add-load
```

### Search throughput and latency

`search-load` performs search only during measurement. Pre-ingest the corpus so
the measured run searches populated user memory.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario search-load --users 20 --duration 30 --seed 0 \
    --preingest --preingest-fraction 1.0 \
    --output out/search-load
```

For a controlled memory-size point, pre-ingest an exact prefix and keep the
query mix fixed independently of corpus size:

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario search-load --users 1 --duration 30 --seed 0 \
    --preingest --preingest-items-per-user 10000 --query-limit 1000 \
    --top-k 20 --output out/search-10k
```

Set `dataset.memories_per_user` in the YAML to at least the largest requested
count. Exact pre-ingest streams only the requested prefix; it does not
materialize the complete per-user corpus in the load generator. `--query-limit`
is valid only for `search-load` and requires that many source memories. Keeping
it constant across runs prevents query-pool growth from becoming a confounding
variable. These controls establish one memory-growth point.

Run the complete experiment with the memory-growth sweep. It executes the same
closed-model `search-load` path at every point, gives every point and repetition
a fresh backend user namespace, and keeps the source corpus and query prefix
identical:

```sh
ltm100 sweep memory-growth --config examples/synthetic.yaml \
    --memory-counts 10,50,100 --queries-per-user 10 \
    --users 4 --duration 30 --warmup 5 --top-k 20 \
    --global-concurrency 16 --repetitions 3 \
    --output out/memory-growth
```

The bundled command above fits the example's default 100-memory synthetic
corpus. For larger points, raise `dataset.memories_per_user` first. The query
count must not exceed the smallest memory point. Before creating output or
calling the backend, the sweep bounded-scans every selected user's source
stream to ensure it can supply the largest requested memory count.
Each repetition is kept under `n_<count>/repeat_<index>/`. The root
`manifest.json` records point status and median/min/max/range across valid
repetitions; `summary.csv` provides one row per repetition. A repetition is
explicitly marked invalid when its completed pre-ingest accounting differs
from the requested point, searches error or reject, or empty results exceed
`--max-empty-rate` (strictly zero by default). Runtime exceptions are recorded
as failed repetitions. Failed and invalid point reports are preserved, and the
sweep exits nonzero rather than silently aggregating them as valid results.

The sweep requires the backend's normal per-user tenancy mapping. A fixed
shared `backend.project_id` is rejected because changing only a producer filter
would not isolate the physical stored corpus between size points.

### Mixed open-model load

`mixed` emits a configurable add/search ratio. Under the open model it is a
lightweight congestion probe for finding the arrival rate at which rejections
begin.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario mixed --users 8 --duration 20 --seed 0 \
    --model open --arrival-rate 5.0 --session-ops 6 \
    --global-concurrency 4 --queue-bound 4 --search-weight 0.8 \
    --preingest --preingest-fraction 0.5 \
    --output out/mixed
```

### MCP transport

The MCP adapter sends measured add/search operations through `add_memory` and
`search_memory`; project setup and teardown remain on REST. The example uses
LongMemEval and requires the `datasets` and `mcp` extras.

```sh
pip install -e ".[dev,datasets,mcp]"

ltm100 run --config examples/memmachine-mcp.yaml \
    --scenario chat-replay --users 20 --duration 30 --seed 0 \
    --output out/mcp-chat
```

### Search expansion and filtering

Supporting REST adapters can forward `expand_context` and metadata filters.
The synthetic dataset writes `metadata.category` when its `categories` option
is enabled.

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario search-load --users 20 --duration 30 --seed 0 \
    --preingest --expand 2 --filter metadata.category=cat_3 \
    --output out/filtered-search
```

### Multi-process client scaling

One asyncio process can saturate a CPU core before a fast server reaches
capacity. `--procs` shards users and whole-run budgets across OS processes.

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 100 --duration 60 --seed 0 \
    --procs 4 --output out/scaled
```

## Common flags

- `--duration SECONDS` or `--ops N`: measured termination condition; one is
  required. Duration stops new dispatch, then drains in-flight requests. A
  closed count-only run executes exactly N operations; setting duration as
  well can stop it earlier. See [termination rules](./load-models.md#termination-and-request-draining).
- `--warmup SECONDS`: run the workload before measurement without consuming
  the measured duration/op budget or recording its requests.
- `--users N`: virtual-user or tenant pool size.
- `--seed N`: reproducible load-shape seed; also selects the synthetic corpus
  and its content-derived queries. Keep it fixed across comparable runs and
  memory-growth points. Seed `0` preserves the original synthetic content.
- `--model closed|open`: scheduling model.
- `--global-concurrency N`: maximum total in-flight operations (`0` = no cap).
- `--arrival-rate F`, `--session-ops N`, `--queue-bound N`: open-model controls.
- `--procs N`: number of load-generator processes.
- `--rampup SECONDS`: stagger closed-model user startup.
- `--preingest`: populate memory before measurement.
- `--preingest-fraction F`: ingest a fraction of each user's complete stream.
- `--preingest-items-per-user N`: stream an exact input-item prefix per user;
  mutually exclusive with `--preingest-fraction`.
- `--query-limit N`: use an exact source-memory prefix as the `search-load`
  query pool, independent of the stored corpus size.
- `--top-k N`: search result count for search-capable scenarios.
- `--expand N`: server-side context expansion, when supported.
- `--filter EXPR`: server-side metadata filter, when supported.
- `--search-weight F`: search fraction for `mixed`.
- `--think SECONDS`: maximum uniform think jitter for `mixed` and
  `chat-replay` operations.
- `--search-every N`: recall cadence in `chat-replay`.
- `--answer-time SECONDS`: mean LLM answer delay in `chat-replay`.
- `--answer-time-variation RATIO`: optional bounded variation around the answer
  delay; `0.3` means +/-30%. Omit it for the legacy Exponential distribution.
- `--user-gap SECONDS`: mean user reading/typing delay in `chat-replay`.
- `--user-gap-variation RATIO`: optional bounded variation around the user
  delay; `0.5` means +/-50%. Omit it for the legacy Exponential distribution.
- `--chat-profile PATH`: group-specific `chat-replay` parameters and session
  controls.
- `--server-metrics`: collect backend-provided server latency metrics.
- `--server-metrics-interval SECONDS`: with `--server-metrics`, write
  fixed-interval server latency breakdowns to `server_metrics_timeseries.csv`.
  Requires `--output`.
- `--raw`: write per-request `raw.ndjson`.
- `--time-series-interval SECONDS`: write client-observed E2E performance to
  `timeseries.csv` in fixed elapsed-time intervals. Requires `--output` but
  does not require `--raw`.
- `--no-delete-on-exit`: preserve per-run backend state.

Run `ltm100 run --help` for the complete and authoritative option list.

## Backend notes

See the [backend support matrix](backend-support.md) for a concise comparison
of workloads, options, observability, and measurement semantics.

### MemMachine REST

[`examples/memmachine.yaml`](../examples/memmachine.yaml) maps each virtual
user to a project under `org_prefix`. By default it uses episodic add/search operations
and supports `--expand`, `--filter`, and `--server-metrics`. Dialogue item
roles are forwarded in `messages[].role`. Optional backend
`add_memory_types`/`search_memory_types` select semantic or combined paths;
see [memory types](memory-types.md) for server requirements and restrictions.

The optional `project_id` and `filter_by_producer` settings support shared-
project isolation experiments. See the isolation-scope section in
[`scenarios.md`](./scenarios.md) before interpreting those results.

### MemMachine MCP

[`examples/memmachine-mcp.yaml`](../examples/memmachine-mcp.yaml) uses MCP for
measured add/search operations and REST for lifecycle management. MCP add
writes all memory types rather than the REST adapter's default episodic-only payload,
so REST and MCP add latency are not a transport-only comparison.

The MCP tools expose neither context expansion nor metadata filtering and do
not accept item metadata or role. Role is intentionally omitted so dialogue
workloads remain runnable; use REST when stored speaker identity is required.
The adapter rejects metadata/filter/expansion combinations rather than silently
dropping settings that change the labeled measurement.

### Mem0 OSS REST

[`examples/mem0.yaml`](../examples/mem0.yaml) maps each virtual user to a
namespaced Mem0 `user_id`. Its default `infer: false` stores each input as one
memory without LLM fact extraction, preserving LTM100's item accounting. Set
`infer: true` to include Mem0's extraction pipeline. Mem0 supports `--filter`
but not `--expand`; dialogue item roles are forwarded in each message.
The `user_id` filter key (including `metadata.user_id`) is reserved for tenant
scope and rejected rather than allowing a filter to override user isolation.

### Supermemory REST

[`examples/supermemory.yaml`](../examples/supermemory.yaml) uses direct
`/v4/memories` creation and `/v4/search` with one container tag per user.
It supports exact-match metadata filters and `top_k` values 1–100. Item role,
producer, and timestamp are retained as provenance metadata. Authentication is
read from an environment variable, not stored in the YAML.

Context expansion is disabled with a warning for this backend, and server-side
metrics are reported as unsupported; client-observed reports remain available.
This path does not benchmark asynchronous document extraction. See the
[Supermemory guide](supermemory.md) for measurement semantics, namespace/cleanup
safety, configuration, and official-server tests.

## Reports

With `--output DIR`, LTM100 writes:

- `summary.json`: metadata and per-operation aggregates. Offered, accepted,
  successful, backend-error, and rejected populations are separate.
  `throughput_ops_s`, `qps`, and latency percentiles contain successful
  requests only. Search `items.empty_rate` is the fraction of successful
  searches returning no items.
  Profiled open-model runs also contain `summary.sessions` with offered,
  admitted, rejected, and rejection-rate counters overall and by group. These
  are session arrivals, separate from request-level queue rejection fields.
- `summary.csv`: flattened summary rows. The overall row omits mixed-operation
  latency because combining add and search latency is ambiguous.
- `raw.ndjson`: per-request records when `--raw` is enabled.
  Profiled records include `group` and `session_id`; the latter identifies a
  reusable per-user conversation lane, not a backend isolation key.
- `timeseries.csv`: fixed-interval client-observed E2E metrics when
  `--time-series-interval SECONDS` is set. Each interval has separate rows for
  add and search plus an `all` throughput row. It records dispatched starts,
  completed requests, successful throughput, errors, rejections, in-flight
  requests at the interval boundary, and successful-request latency
  statistics. Empty intervals remain in the file so stalls are visible;
  overall latency is blank because add and search distributions are not mixed.
- `server_metrics.csv` and `server_metrics_raw.json`: server-side latency
  deltas when `--server-metrics` is enabled and supported by the adapter.
- `server_metrics_timeseries.csv`: adjacent-snapshot server histogram deltas
  when `--server-metrics-interval SECONDS` is set. Each interval has separate
  rows for MemMachine's add phases, search phases, and add/search HTTP paths,
  with count, mean, p50, p90, and p99 latency. A failed scrape marks the
  affected interval instead of combining it with a later successful scrape.

For example, a five-second time series can be collected without retaining
every per-request record:

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario search-load --users 20 --duration 60 --preingest \
    --global-concurrency 20 --time-series-interval 5 \
    --server-metrics --server-metrics-interval 5 \
    --output out/search-timeseries
```

Request latency starts immediately before the backend client call and ends
when that call returns or raises. Scenario delays and waiting for an LTM100
global-concurrency slot are excluded; network time and all server-side work
after dispatch are included. A request's latency is assigned to its completion
interval, while `started`, `completed`, and `in_flight_end` expose backlog and
stall behavior. Percentiles use successful completions only.

`meta` records the whole-run configuration, server build, timestamps that
bracket the complete run lifecycle, and the exact
`measurement_started_at`/`measurement_ended_at` boundaries. `--warmup N` runs
the selected workload for N seconds before measurement; those requests update
backend state but do not consume `--duration`/`--ops` or appear in summaries,
raw output, or the time series. Server metrics bracket the same measured
window. With multiple processes, time-series runs synchronize workers at both
measurement boundaries, as do warm-up and server-metrics runs.

Periodic server snapshots run on a dedicated sampler thread so the load
generator's asyncio loop does not wait between scrapes. Sampling uses fixed
monotonic deadlines rather than sleeping relative to the previous scrape; if
a scrape itself overruns one or more deadlines, those boundaries are skipped
instead of issuing catch-up scrapes. The final partial interval is retained.
Server percentiles remain estimates interpolated from the server's Prometheus
histogram buckets. Use an interval long enough to collect a meaningful sample
count; 5--10 seconds is a practical starting point.

Server-side resource utilization such as CPU, memory, storage, and network is
outside LTM100's client report and should be collected from the system under
test.

## Cleanup

Per-run backend state is deleted on exit unless `--no-delete-on-exit` is set.
To delete user state without running a benchmark:

```sh
ltm100 cleanup --config examples/memmachine.yaml --users 50
```

Lifecycle behavior is backend-specific. For example, a MemMachine project list
may remain eventually consistent briefly after a confirmed delete response.

## Reproducible runs

- Keep the YAML, complete CLI command, LTM server build, and hardware topology
  with each result.
- Use the same dataset split, seed, user count, and pre-ingest fraction when
  comparing systems.
- Confirm that search runs return non-empty results; a low error rate alone
  does not prove that the corpus was populated correctly.
- Increase `--procs` only after checking whether the load-generator process is
  the bottleneck.
- Treat REST, MCP, extraction-enabled, and extraction-disabled paths as
  different workloads unless their server-side behavior is equivalent.
