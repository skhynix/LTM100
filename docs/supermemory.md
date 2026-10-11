# Supermemory REST backend

Select `backend.name: supermemory`; see
[`examples/supermemory.yaml`](../examples/supermemory.yaml). The adapter uses
the existing async REST transport, with no SDK dependency or changes to the
runner, scenarios, profiles, workload parameters, or reporting.

## API and measurement semantics

| LTM100 operation | Supermemory request | Behavior |
|---|---|---|
| Setup | None | Containers are created lazily; existing data is not reset |
| Add | `POST /v4/memories` | Direct creation, one returned ID per input |
| Search | `POST /v4/search` | `searchMode: memories`, scoped by `containerTag` |
| Cleanup | `DELETE /v3/container-tags/{tag}` | Delete only the requested users' containers |

This backend measures **direct memory creation and retrieval**, not the
asynchronous `/v3/documents` extraction pipeline. It does not poll for background
ingestion or count a queued document acknowledgement as a completed memory add.
It explicitly disables query rewriting and result aggregation. `rerank` can be
configured independently and defaults to `false`.

`add_batch_size` is 1 by default and accepts 1–100. Larger calls are split into
sequential batches, preserving input order and per-item metadata. A workload add
is still one LTM100 operation even when the adapter sends multiple HTTP requests.
The adapter checks returned ID counts; those counts do not prove that the server
retains that many distinct memories indefinitely (e.g. deduplication or expiry).

Search `top_k` maps directly to `limit`; supported values are 1–100. Values outside
this range fail rather than being silently clamped. `threshold` is explicit,
defaults to 0.0 (no similarity cutoff), and accepts 0–1; Supermemory's own default
is 0.6. Record these YAML options when interpreting benchmark results.

## Authentication and isolation

Set the key in an environment variable, not in the YAML:

```sh
export SUPERMEMORY_API_KEY=<your-api-key>
ltm100 run --config examples/supermemory.yaml \
    --scenario search-load --users 4 --ops 100 --top-k 20 \
    --preingest --output out/supermemory
```

`api_key_env` selects the variable name (default `SUPERMEMORY_API_KEY`). The key
is resolved separately in each worker and sent only as a bearer header; it is
not a backend configuration value in reports. A missing key fails early.

Each user maps to `{user_prefix}_{sha256(full_user_id)}`. A full digest handles
Unicode, long IDs, and memory-growth namespaces without sanitization collisions.
`user_prefix` must contain 1–35 letters, digits, underscores, colons, or hyphens,
keeping the complete tag inside the server's 100-character limit.

Use a **fresh prefix for each independent run**, especially for parallel runs.
The same prefix/user maps to the same server state. Setup does not clear old
data. Default cleanup deletes the entire mapped container, including any older
data sharing that tag; use only a dedicated benchmark namespace. No
organization-wide reset is performed. `--no-delete-on-exit` retains the data,
and `ltm100 cleanup` uses the same mapping. A cleanup key needs owner/admin
permission; 401/403/429/5xx are surfaced, while an already-absent container's 404
is treated as successful cleanup.

## Metadata and unsupported features

- Original content and item metadata are retained. Speaker role, producer, and
  timestamp are stored as `ltm100_role`, `ltm100_producer`, and
  `ltm100_timestamp` metadata. These are provenance fields, not a native chat
  message schema. Input metadata using these reserved keys is rejected before
  any batch is written.
- `--filter metadata.key=value` maps to an exact-match metadata `AND` filter.
  The filter never replaces the user's container tag. Values remain strings;
  this syntax does not express numeric predicates or compound conditions.
- `--expand` is unsupported: expansion is disabled **only for this backend**,
  with one warning per adapter instance. Results are ordinary memory hits, not
  neighboring episodes. The requested CLI value is not evidence that expansion
  occurred; keep `--expand 0` for comparison runs.
- Server-side metrics and server metrics time series are unsupported. The
  existing CLI warns and reports `status: unsupported`; a requested server
  trace has a header-only CSV, not fabricated measurements. Client raw records
  and E2E `timeseries.csv` work normally.
- Per-user container tags are the only isolation strategy in this adapter;
  backend-specific shared-project/producer-filter options do not apply.

Synthetic data supports `add-load`, `search-load`, and `mixed`. For `chat-replay`,
replace only the dataset section with a LongMemEval configuration. Existing
closed/open load controls, profiles, timing variation, warm-up, multiprocess
execution, and memory-growth sweeps remain unchanged.

## Official-server integration tests

Run against a **disposable official local server** with an owner/admin key:

```sh
export LTM100_SUPERMEMORY_E2E_URL=http://127.0.0.1:6767
export SUPERMEMORY_API_KEY=<your-local-api-key>
pytest -q tests/test_supermemory_e2e.py
```

The opt-in tests use unique prefixes and exercise direct add/search, metadata
filters and provenance, tenant isolation, cleanup, the four original scenarios,
closed/open profiled chat, multiprocess execution, warm-up, raw/client time
series, unsupported server metrics, and isolated memory-growth points. Without
the explicit URL they are skipped. API-contract unit tests run without any
external server. Coding-agent additionally exercises this adapter through
API-contract tests; live coding-agent verification currently uses MemMachine.

Local verification can use a deterministic OpenAI-compatible embedding endpoint
instead of a paid model. That exercises the official server's storage and search
paths but does **not** validate retrieval quality, production capacity, hosted
rate limits, or document extraction. Never interpret those timings as production
Supermemory performance.

The integration suite was verified with the official Linux x64 release
`server-v0.0.8` and a local deterministic 32-dimensional embedding endpoint.
The server used its embedded storage engine, not a replacement API server.
The adapter requires the direct-memory, memory-search, and container-deletion
endpoints above; deployment versions and key permissions must support them.
No hosted account or production-model performance was verified.

Official references:

- [Direct memory API](https://supermemory.ai/docs/api-reference/content-management/create-memories-directly)
- [Memory search API](https://supermemory.ai/docs/api-reference/recall-search/search-memory-entries)
- [Container deletion](https://supermemory.ai/docs/api-reference/container-tags/delete-container-tag)
- [Local server configuration](https://supermemory.ai/docs/self-hosting/configuration)
