# LTM100

LTM100 is an end-to-end multi-user **load benchmark** for Long-Term Memory
(LTM) systems. It drives virtual users through realistic and focused memory
workloads and reports client-observable throughput, latency, rejection, and
error metrics.

LTM100 measures **load, concurrency, and scalability behavior**, not retrieval
or answer quality. It does not compute precision, recall, or MRR.

Current release: **v0.4.3**

## Highlights

- Closed and open load models for fixed-concurrency and arrival-driven tests.
- Four scenarios: `chat-replay`, `add-load`, `search-load`, and `mixed`.
- Weighted chat workload profiles with group-specific recall, timing, and
  closed/open session controls.
- Bounded admission and rejection accounting for overload experiments.
- Multi-process load generation so the benchmark client can scale beyond one
  event-loop core.
- Optional pre-ingest, ramp-up, raw request records, fixed-interval E2E time
  series, and server-side latency metrics.
- Repeated memory-growth sweeps with fixed query workloads and isolated corpus
  points.
- Pluggable datasets and backend adapters.
- JSON and CSV reports with successful, error, and rejected traffic separated.

## Install

```sh
pip install -e ".[dev]"
```

Install optional dependencies when needed:

```sh
pip install -e ".[dev,datasets]"      # LongMemEval / Nebius via Hugging Face
pip install -e ".[dev,datasets,mcp]"  # LongMemEval + MCP transport
```

LTM100 requires Python 3.10 or later. The `ltm100` command is the primary
entry point; `python -m ltm100.cli` is equivalent when the console script is
not on `PATH`.

## Quick start

Point an example YAML at a running LTM endpoint, then choose a scenario and a
termination condition:

```sh
ltm100 run --config examples/synthetic.yaml \
    --scenario add-load --users 20 --duration 30 --seed 0 \
    --output out/add-load
```

For a chatbot-style workload using LongMemEval:

```sh
ltm100 run --config examples/memmachine.yaml \
    --scenario chat-replay --users 10 --duration 60 --seed 0 \
    --output out/chat-replay
```

See the [running guide](docs/running.md) for configuration, workload commands,
load controls, backend limitations, reports, and cleanup. The
[example matrix](examples/README.md) shows the intended use of each YAML file.

## Workloads

| Scenario | Purpose | Dataset requirement |
|---|---|---|
| `chat-replay` | Replay chatbot recall and conversation ingestion | Structured `turn_stream` such as LongMemEval |
| `add-load` | Measure pure memory-ingest throughput | Any dataset |
| `search-load` | Measure search throughput and latency | Any dataset; pre-ingest recommended |
| `mixed` | Generate a configurable add/search mixture | Any dataset; pre-ingest recommended |

Every scenario runs under both load models:

- `closed`: a fixed number of users repeatedly execute their plans.
- `open`: sessions arrive through a Poisson process and exercise the bounded
  queue and rejection policy.

LongMemEval supports all four scenarios. The bundled synthetic dataset supports
the three scenarios that do not require structured dialogue turns.

## Adapters

Datasets:

- **LongMemEval** — local JSON or Hugging Face loading, with structured
  user/assistant turns for `chat-replay`.
- **Synthetic** — deterministic, download-free data for load probes.
- **Nebius** — structured coding-agent trajectories and corpus streams for
  generic add/search workloads; see the [dataset guide](docs/nebius.md).

Backends:

- **MemMachine REST**
- **MemMachine MCP** with REST lifecycle management
- **Mem0 OSS REST**
- **Supermemory REST** with direct memory creation

See the [backend support matrix](docs/backend-support.md) for workload, option,
and observability support across adapters, and the
[running guide](docs/running.md) for configuration details.
See the [Supermemory guide](docs/supermemory.md) for its API scope and limitations.

## Reports

With `--output DIR`, a run writes `summary.json` and `summary.csv`. Use
`--time-series-interval SECONDS` for client-observed E2E `timeseries.csv`,
`--raw` for per-request `raw.ndjson`, and `--server-metrics` for
backend-provided server latency breakdowns when the selected adapter supports
them. Add `--server-metrics-interval SECONDS` to trace those server-side
breakdowns over the measured window.

Successful throughput and latency are reported separately from backend errors
and queue rejections, so overload cannot inflate QPS or lower service-latency
percentiles. See [Reports](docs/running.md#reports) for the output contract.

## Documentation

- [Running guide](docs/running.md) — configuration, commands, flags, reports,
  and cleanup.
- [Backend support](docs/backend-support.md) — shared features, adapter limits,
  and comparison semantics.
- [Memory types](docs/memory-types.md) — opt-in backend type selection and defaults.
- [Scenario guide](docs/scenarios.md) — per-scenario data flow and behavior.
- [Load models](docs/load-models.md) — closed/open scheduling and congestion.
- [Workload patterns](docs/workload-patterns.md) — fixed-delay chat-replay
  example with recorded request-count graphs.
- [Example configurations](examples/README.md) — YAML selection guide.
- [Design](DESIGN.md) — adapter contracts, lifecycle, metrics, and roadmap.

## Development

```sh
pytest -q
ruff check .
```

## Versioning

Releases use `vMAJOR.MINOR.PATCH` tags. During 0.x, a minor bump marks a
meaningful tested milestone; breaking changes bump the minor version.

```sh
git tag v0.4.3
git push origin v0.4.3
```

## License

[Apache License 2.0](LICENSE)
