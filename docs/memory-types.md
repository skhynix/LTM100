# Configurable memory types

Memory types are backend options, independent of scenario, dataset, and load
model. They do not change operation order, timing, request budgets, or latency
measurement. Existing YAML files require no edits.

## Defaults and support

| Adapter | Options omitted | Explicit type selection |
|---|---|---|
| MemMachine REST | Episodic-only add and search, as before | `episodic`, `semantic`, or both, independently per operation |
| MemMachine MCP | Existing server all-types calls; adapter returns long-term episodic results | Rejected; tools expose no selection argument |
| Mem0 OSS REST | Existing `infer` behavior, default `false` | Rejected; `infer` is not mapped to MemMachine memory types |
| Supermemory REST | Existing direct-memory behavior | Rejected; no equivalent selection in this adapter |

Only MemMachine REST accepts `add_memory_types` and `search_memory_types`.
Each defaults independently to `[episodic]`; omission never delegates selection
to the server's all-types default. Lists must be non-empty, unique, and contain
only the exact strings `episodic` and `semantic`. Explicit null, strings in
place of lists, unknown types, and duplicate entries fail validation. No
unsupported selection is silently ignored or translated to another backend's
inference setting.

```yaml
dataset:
  name: synthetic
backend:
  name: memmachine
  base_url: http://localhost:8080
  add_memory_types: [episodic, semantic]
  search_memory_types: [episodic]
```

This adds to both memory paths but measures episodic search. Omitting
`search_memory_types` has the same effect. Conversely, requesting semantic
search does not automatically enable semantic writes: select matching write
types or supply server data separately.

A complete opt-in example is
[`examples/memmachine-memory-types.yaml`](../examples/memmachine-memory-types.yaml).
To use chat-replay, put the same backend options in a LongMemEval config; no
new scenario flags are needed.

## Server readiness and unsupported combinations

If either operation selects semantic memory, the adapter performs a read-only
`GET /api/v2/config/memory/semantic` check before workload setup. The server
must report boolean `enabled: true`. Disabled, inaccessible, unsupported, or
malformed responses fail the run before provisioning or workload cleanup.
For MemMachine versions that gate the config API, enable it when starting
your disposable/test server, for example `memmachine-server --with-config-api`.
Deployment access controls still apply. LTM100 never enables semantic memory,
changes server settings, or starts models on the operator's behalf.

Episodic-only runs perform no additional config request and work with servers
without the config API. Successful semantic preflight is cached per adapter
connection; it verifies enabled configuration, not extraction health or feature
readiness. Standalone `cleanup` does not require semantic readiness.

This adapter currently rejects:

- Either semantic add or search with `filter_by_producer: true`. Episodic
  producer filtering must not be assumed to isolate semantic features.
- Semantic search with `--filter`/`QueryItem.filter` or
  `--expand`/`QueryItem.expand_context`. These options remain available for
  episodic-only search, including when writes select both types.

A shared `project_id` without producer filtering remains one intentional
shared memory pool, not per-user isolation. For isolated semantic evaluation,
use the default project-per-user mapping and a fresh `org_prefix` per run.

## Results and interpretation

Episodic parsing and its metadata are unchanged. Semantic features become
`ResultItem`s with:

- `content`: `feature_name: value`;
- `uid`: native `metadata.id`, or null if absent;
- `score`: null; the API does not provide a comparable ranking score;
- `metadata.ltm100_memory_type`: `semantic`;
- `metadata.semantic_feature`: the full native feature, including category,
  tag, set ID, and available citations/other metadata.

Malformed semantic entries raise operation errors instead of becoming empty
successful searches. When both types are requested, the result list contains
long-term episodic results followed by semantic results. The adapter neither
reranks across types nor deduplicates or truncates the combined list. The
server can apply `top_k` per memory type; the combined count can exceed it.
Search `n_items` and empty rate describe the combined returned list.

ADD `n_items` still counts returned input-message IDs. Pre-ingest statistics
still count submitted input items. Neither describes the number of generated
semantic features. Semantic add also retains source episodes in the server's
raw storage; semantic-only does not mean that all underlying episode storage
is bypassed.

Semantic extraction is asynchronous. Add/pre-ingest acknowledgement does not
promise searchable semantic features yet. LTM100 inserts no automatic readiness
polling, sleeps, or background-work drain into the workload. Existing warm-up
and timing controls retain their meaning. Request E2E latency, measured-window
server metrics, and run completion are not semantic-extraction completion or
background-processing capacity measurements.

## Reports

MemMachine REST reports include `meta.memory_types`:

```json
{
  "requested": {"add": ["episodic", "semantic"], "search": null},
  "resolved_request_types": {"add": ["episodic", "semantic"], "search": ["episodic"]}
}
```

Null in `requested` means the YAML option was omitted, not that a null option
is accepted. Resolved types describe the request payload, not independently
observed server execution. Settings are whole-run values and do not change
with process count; endpoint/auth settings are not copied into this metadata.

## Verification

```sh
pytest -q tests/test_memory_types.py
```

Tests cover existing YAML/default payload compatibility, independent type
combinations, malformed results, invalid settings, unsupported adapters,
preflight failures without provisioning/deletion, and cleanup independence.

Live tests need a disposable semantic-enabled MemMachine server with a
controlled LLM that extracts `LTM100_FACT:<value>` into a `test_fact` feature.
An embedding mock plus an empty chat response cannot validate this path.

```sh
LTM100_SEMANTIC_E2E_URL=http://127.0.0.1:8080 \
  pytest -q tests/test_memory_types_e2e.py
```

They verify actual extraction, feature persistence/search, selected index
paths, tenant isolation, cleanup, all four workloads under closed/open models,
CLI pre-ingest/warm-up/reporting, and multiprocessing.
Only the test fixture polls for extracted facts outside workload measurement;
the production runner never does. Mock models validate functionality, not
retrieval quality or production performance.

Disabled-server preservation can be checked separately, against a disposable
server with semantic memory disabled and its config API available:

```sh
LTM100_SEMANTIC_DISABLED_E2E_URL=http://127.0.0.1:8080 \
  pytest -q tests/test_memory_types_disabled_e2e.py
```

This provisions an episodic corpus first and verifies a rejected semantic run
does not delete that pre-existing project or alter its contents.
