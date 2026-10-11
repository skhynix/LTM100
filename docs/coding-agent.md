# Coding-agent replay

`coding-agent` models LTM request patterns around recorded agent activity.
It consumes the dataset's structured `task_stream`, currently provided by
[Nebius](nebius.md), without changing `chat-replay` or the generic corpus
workloads. It never executes tools, patches, repositories, or an agent model.
LTM search results do not alter subsequent recorded actions. Requests are
awaited, so backend latency affects workload pacing.

## Run and configure

Dataset loading, source length, seed-based user assignment, and backend
selection use the same adapter YAML/CLI workflow as LongMemEval:

```sh
ltm100 run --config examples/nebius.yaml --scenario coding-agent \
    --agent-profile examples/agent-profile.yaml \
    --users 10 --duration 300 --seed 0 --top-k 20 \
    --raw --time-series-interval 1 --output out/coding-agent
```

The optional [agent profile](../examples/agent-profile.yaml) configures one
uniform policy. With no profile, defaults are user-facing hook replay,
zero modeled delays, and task repetition. `--chat-profile` remains exclusive
to `chat-replay`; it does not configure coding-agent users or concurrency.
All settings are validated before backend mutation. Unknown profile keys are
rejected, and effective settings appear in `summary.json`'s `meta.agent_profile`.

| Setting | Default | Meaning |
|---|---|---|
| `version` | 1 | Profile schema version |
| `integration` | `hook` | `hook`, `mcp`, or `hybrid` |
| `hook_scope` | `user-facing` | `user-facing` or `all-steps`; inactive for MCP-only |
| `search_probability` | 0.25 | MCP selection probability before ordinary assistant steps |
| `add_probability` | 0.25 | MCP selection probability after recorded ordinary tool results |
| `agent_step_time` | 0 | Modeled generation seconds per assistant message |
| `tool_time` | 0 | Modeled seconds before each recorded ordinary tool result |
| `tool_times` | `{}` | Tool-name overrides for `tool_time`; unknown ordinary names allowed |
| `task_gap` | 0 | Closed-model seconds between tasks, including wrap-around |
| `agent_step_time_variation`, `tool_time_variation`, `task_gap_variation` | Unset | Same convention as chat-replay: unset = Exponential(mean), ratio v = Uniform(mean × (1−v), mean × (1+v)); 0 = fixed |
| `task_repeat` | `true` | Repeat a user's assigned tasks after exhaustion |
| `chunk_chars`, `query_chars` | 3000 | Positive character limits for stored chunks and queries |

Probabilities and variations are ratios in [0, 1]; durations must be finite and
non-negative. Both MCP-only probabilities cannot be zero. Timing settings are
**synthetic assumptions**, not source timestamps; command timeout arguments do
not supply actual tool durations. Multiple tool observations are modeled in
source order with separate sequential waits, not reconstructed parallel tool
execution. LTM calls are synchronous from the replay's perspective, even when a
server starts background ingestion. Background client adds are not modeled.

## Hook and MCP semantics

| Policy | Search | Add |
|---|---|---|
| Hook / user-facing | Before the first source user request | First request and recorded terminal assistant response |
| Hook / all-steps | Before each source user/tool input | That input and every recorded assistant output |
| MCP-only | Probabilistically before each non-terminal assistant step | Probabilistically after each ordinary tool result |
| Hybrid | Hook searches plus MCP searches | Hook adds plus MCP adds; no automatic deduplication |

For one initial request, one ordinary tool action/result, and a terminal
`finish`, with MCP probabilities set to 1 and no chunking:

- User-facing hook: SEARCH request → ADD request → agent wait → tool wait →
  final agent wait → ADD final response.
- All-steps hook: SEARCH request → ADD request → agent wait → ADD action →
  tool wait → SEARCH observation → ADD observation → final agent wait →
  ADD final response.
- MCP-only: SEARCH request → agent wait → tool wait → ADD completed step →
  final agent wait. Terminal `finish` causes no MCP search/add.
- Hybrid: combines the corresponding hooks and selective MCP calls at these
  same source positions. Calls do not recursively trigger hooks.

System instructions are excluded. The first source `user` message is the task
request; later `user` messages may be framework feedback and are not treated
as new human prompts. They are included by all-steps hooks and can become the
known input for a later MCP search, but user-facing hooks skip them.

A search uses the initial request and latest known input, bounded by
`query_chars`; it never uses future observations or the final answer. MCP adds
bundle the issuing assistant action with that result and use role `tool`.
For multiple calls, each recorded result has its own selection decision and
bundle; the issuing action retains its complete call list. Hook payloads retain
source roles. Tool arguments are serialized deterministically as text. A
terminal response is stored once without duplicating the `finish` wrapper.
Empty outputs add no items. Long payloads become individual chunk ADDs, each
of which consumes one operation budget unit. Search selection occurs once
per source input/step, not once per chunk.

Interrupted attempts retain recorded activity and waits, without generating
missing tool results or a final answer. `resolved` and `exit_status` describe
the original coding attempt, not LTM errors or retrieval quality.

**MCP here names a synthetic memory-use policy**, not observed decisions by
an agent and not an enforced network protocol. The backend YAML independently
selects REST or MCP transport. Using this policy over REST does not benchmark
MCP protocol overhead. Backend role/filter/top-k/observability limitations
remain as described in the [support matrix](backend-support.md).

## Task boundaries and load models

User mapping and finite dataset streams remain unchanged; see
[Nebius assignment](nebius.md#user-mapping-and-repetition). The scenario owns
repetition and never resets retained backend state between tasks. Repeated
content can grow or deduplicate according to backend behavior.

- **Closed:** one sequential task lane per user, following the assigned order.
  `task_gap` applies before each subsequent task, including the first task after
  wrap-around. There is no gap before the initial task. Ramp-up and global
  concurrency retain their existing meanings.
- **Open:** each eligible admission consumes exactly one next assigned task.
  Assignment uses admission order, not completion order, and persists across
  arrivals for that user. `session_ops` may truncate the task but never crosses
  its boundary. The next arrival advances to the next task even if the previous
  one was truncated. `task_gap` is inactive; arrival rate controls session starts.
  Multiple tasks may be active for one backend user, as with ordinary open runs.
- **Finite replay:** `task_repeat: false` stops closed users after their task
  pools are exhausted. Duration is an upper bound, so closed runs may finish
  earlier. Pure count-based closed runs are rejected because the exact budget
  may be unreachable. Core configurations with both duration and ops are
  allowed; whichever bound is reached first limits replay. In open runs, arrivals targeting an exhausted pool do
  not enter offered/admitted/rejected session counters; the run continues to its
  duration deadline. Exhaustion is not a congestion rejection.

For a user's A/B pool, closed replay follows A → B → A; successive open
admissions also select A, B, A, regardless of their completion order. Task
occurrence starts at 0. Warm-up and measurement restart independent task and
RNG cursors while keeping warm-up backend writes, consistent with existing
warm-up behavior. Reproducibility covers source assignment and planned policy/
timing, not wall-clock scheduling or identical request counts in duration runs.
Changing user count changes assigned pools. Process count does not change
per-user plan prefixes; pooled count budgets need not stop every user at the
same position.

## Timing, termination, and reports

Agent/tool/task waits are non-request timeline events. They do not occupy
request concurrency slots, consume `--ops`/`--session-ops`, create rejection
records, or enter request latency/QPS. Wait-only steps and trailing task time
are retained; zero-duration waits cooperate with the event loop. Deadlines
stop new requests and waits, while requests already executing may finish.
Reaching a count budget can truncate remaining task activity. A pure count-based
MCP-only closed run is rejected if any assigned user's pool has no eligible call
positions. Unexpected coding-agent plan errors fail the run rather than
producing a successful partial report. Open admission/preparation errors stop
further dispatch and drain all existing sessions before the error reaches the
caller, so backend teardown cannot overlap surviving workload sessions.
Already dispatched requests finish; pending waits are bounded by the deadline.

Existing summary, raw, client time series, server metrics, pre-ingest, cleanup,
and multiprocess paths are reused. Pre-ingest still uses the dataset's generic
`memory_stream` projection; it does not simulate hook/MCP calls. Server metrics
are collected only where supported and use the existing measured window.

With `--raw`, coding-agent requests additionally carry `context`:

```json
{"trajectory_id":"trace-0","instance_id":"issue-0","task_occurrence":0,
 "step":2,"source_exit_status":"submit","source_resolved":false,
 "origin":"mcp"}
```

ADD context also includes `chunk_index`. This context stays in reports, not
backend metadata. Existing workloads omit it and retain their raw schema.
Waits have no fake request records. LTM operation latency measures the awaited
backend call; it excludes simulated agent/tool waits and is not task completion
time, coding success, or answer quality.

Keep source `length` manageable: tasks are loaded lazily with one lookahead per
user and active task, but Parquet row-group decoding and source parsing still
use client resources. Monitor client saturation when interpreting performance.

## Verification

Offline tests cover policy order and payloads, source/tool edge cases,
reproducible selection, independent timing randomness, task cycling and
admission, wait-only activity, deadlines, exact budgets, warm-up, multiprocessing,
open admission/preparation failures, cancellation, and all four adapter
contracts:

```sh
pytest -q tests/test_coding_agent.py tests/test_coding_agent_runner.py \
    tests/test_coding_agent_backends.py
```

Live tests require an explicitly configured disposable MemMachine server:

```sh
LTM100_MEMMACHINE_E2E_URL=http://127.0.0.1:8080 \
    pytest -q tests/test_coding_agent_e2e.py
```

They exercise five policy combinations under both models, each with one/two
processes, pre-ingest, warm-up, raw/client/server reports, isolated storage,
and cleanup. Server HTTP add/search deltas are compared with measured request
counts. Additional failure-path tests inject a later source-read error during
warm-up and measurement, verify the executing request finishes, and check
that cleanup and transport closure leave no surviving session calls.
Set `LTM100_NEBIUS_E2E_PATH` to use local official records; the tests
select two trajectories. Default fixtures are authored, not bundled source data.
A standalone SQLite/mock-embedding setup verifies functionality and report
alignment, not production capacity or retrieval quality.
