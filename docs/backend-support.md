# Backend support

This describes the **current LTM100 adapters**, not each product's full feature
set. A supported request still requires a compatible server/API version.
MemMachine REST is the reference path; its MCP adapter is listed separately
because the measured operations and supported fields differ.

## Common workload and client features

All four adapters use the same runner and support:

- `add-load`, `search-load`, and `mixed` with Synthetic, LongMemEval, or Nebius;
  `chat-replay` with structured dialogue data such as LongMemEval. Synthetic
  has no dialogue turns and cannot run `chat-replay` on any backend.
  Nebius likewise exposes agent tasks rather than chatbot turns; its generic
  corpus workloads do not replay hook/MCP integration policies.
- Closed/open load models, user counts, global request concurrency, open-model
  arrival/session/queue controls, and closed-model ramp-up.
- Chat profiles: group-specific `top_k`, closed `concurrent_sessions`, open
  `max_sessions_per_user`, recall cadence, think time, answer time, user gap,
  and bounded timing variation. Backend search limits still apply.
- Pre-ingest by fraction or exact input-item count, warm-up exclusion, and
  multiprocess execution. `--query-limit` fixes the `search-load` query pool.
- JSON/CSV summaries, `--raw` request records, and client-observed
  `--time-series-interval` output, including error/rejection accounting.
- Isolated memory-growth sweeps with fixed queries. These use closed
  `search-load` and fresh user namespaces; a fixed shared `project_id` is
  rejected. Exact pre-ingest counts describe **input items**, not independently
  verified retained server memories.
- Default cleanup, `--no-delete-on-exit`, and the `cleanup` command, subject to
  the backend-specific deletion behavior below.

## Adapter-specific capabilities

“Rejected” means the adapter raises an error; it does not silently discard the
requested feature. “Disabled” means the run continues with an explicit warning.
Search-option errors become operation errors; metadata rejection during
pre-ingest aborts the run.

| Capability | MemMachine REST | MemMachine MCP | Mem0 OSS REST | Supermemory REST |
|---|---|---|---|---|
| YAML backend name | `memmachine` | `memmachine-mcp` | `mem0` | `supermemory` |
| Add path | Episodic by default; optional semantic or both | All memory types via MCP tool | Raw storage by default (`infer: false`) | Direct memories; no document extraction |
| Per-user scope | One project per user | One project per user | Namespaced `user_id` | Namespaced hashed `containerTag` |
| Memory-type selection | Independent add/search lists; episodic default | Rejected | Rejected | Rejected |
| Shared project + producer-filter isolation | Episodic-only | Rejected | Not implemented | Not implemented |
| Search `top_k` | Forwarded to server | Forwarded to tool | Forwarded to server | 1–100; other values rejected |
| Context expansion (`--expand`) | Episodic search only | Rejected | Rejected | Disabled with a warning |
| Search filters (`--filter`) | Server filter expression; episodic search only | Rejected | String equality, `key=value` | String equality, metadata `AND` filter |
| Item metadata | Values converted to strings | Non-empty metadata rejected | Per-item metadata | Preserved; `ltm100_*` provenance keys reserved |
| Dialogue role | `messages[].role` | Omitted | `messages[].role` | `metadata.ltm100_role` |
| Producer / timestamp | Native message fields | Producer → `user_id`; timestamp omitted | Producer → message `name`; timestamp → metadata | `ltm100_producer` / `ltm100_timestamp` metadata |
| Wire batching | Configurable; default 50 items/request | One tool call/item | One HTTP request/item | Configurable 1–100; default 1 item/request |
| Server latency breakdown + server time series | Supported if Prometheus endpoint is available | Unsupported | Unsupported | Unsupported |
| Automatic server version in run metadata | Health endpoint | No probe | No probe | No probe |
| Exposed authentication setting | None | None | Optional `X-API-Key` | Bearer key from `api_key_env` |

For Mem0, `user_id` is reserved for tenant scope and cannot be supplied as a
filter key, including `metadata.user_id`. Supermemory's three reserved item
metadata keys are `ltm100_role`, `ltm100_producer`, and `ltm100_timestamp`.

Filter syntax is **not portable**. MemMachine forwards the expression unchanged
(for example, `m.category = 'cat_3'`, depending on the server grammar), whereas
Mem0/Supermemory translate `metadata.category=cat_3` into their own request
formats. Do not reuse one backend's filter command without checking its grammar.

See [memory types](memory-types.md) for semantic preflight, result counting,
asynchronous extraction, and incompatible settings.

## Interpreting comparisons

- The same scenario does not imply identical server work. MCP adds include
  semantic processing; Mem0 `infer: true` includes fact extraction and may
  produce zero or multiple memories per input; Supermemory uses direct creation
  rather than asynchronous document ingestion. Use the documented default
  paths when comparing raw add/search workloads and retain the YAML options.
- Role forwarding and provenance metadata are different representations. MCP
  dialogue remains runnable but does not retain role/timestamp fields through
  this adapter. Supermemory stores speaker identity as metadata, not a native
  chat-message role.
- Unsupported server metrics warn and report `status: unsupported`; a requested
  server time-series CSV has headers only. Client summaries/raw/E2E time series
  remain available. Record server versions separately where there is no probe.
- Cleanup is scoped, not a snapshot/restore mechanism. MemMachine leaves a
  pre-existing shared project intact; deletion HTTP failures are logged at
  debug level in its REST/MCP adapters. Mem0/Supermemory surface deletion errors;
  Supermemory treats an absent container's 404 as already cleaned up. Use a
  dedicated run namespace to avoid mixing or deleting older data. MemMachine
  shared-project multiprocess runs require `--no-delete-on-exit`.

Configuration and command details: [running guide](running.md),
[scenario/isolation guide](scenarios.md),
[Supermemory guide](supermemory.md), and
[example matrix](../examples/README.md).
