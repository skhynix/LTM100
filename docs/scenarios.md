# Scenarios

This document describes what each LTM100 scenario tests and exactly how a
virtual user behaves under it: how users are driven, how many run at once,
the concurrency model, and the `add`/`search` operations issued. Read it
alongside [`DESIGN.md`](../DESIGN.md) for the contracts and the
[README](../README.md) for copy-pasteable run commands.

LTM100 is backend-pluggable: scenarios talk to an `LTMClient` adapter, never
to a specific server. This document is therefore written at the adapter
contract level. Backend-specific details (MemMachine's REST endpoints,
tenancy key, request bodies) are isolated to the last section,
[MemMachine backend specifics](#memmachine-backend-specifics).

## Shared behavior

All scenarios share the same primitives. Understanding these first makes the
per-scenario differences small.

### Users and tenancy

A virtual user is **one asyncio coroutine** in a process (not a container).
`--users N` spawns N coroutines, each identified by a `UserId` string. By
default all coroutines run in a single process; `--procs N` shards them across
N OS processes (each driving a disjoint slice of the users) so a fast server
is not bottlenecked on one event-loop core.

Per-user isolation is the **adapter's** responsibility: it maps a `UserId`
to whatever tenant key the backend uses, so that a user only ever searches
its own memories. The adapter's `setup` provisions per-user tenants and
`teardown` removes them.

When `--users N` exceeds the dataset's unique samples, samples are
replicated, so N is the driven virtual-user count, independent of dataset
size.

### Concurrency

Concurrency is controlled on two layers:

1. **Per-user in-flight = 1 (fixed).** Each user emits one request at a time
   and waits for the response before sending the next. There is never more
   than one in-flight request per user.
2. **`--global-concurrency C` (optional, 0 = no cap).** A global bound on the
   total number of in-flight requests across all users, enforced with an
   `asyncio.Semaphore`. Even with N users firing simultaneously, at most C
   requests execute at once.

### The add and search operations

A scenario emits `Op`s of two kinds, carried over the `LTMClient` contract:

- **ADD** — `client.add(user, items: list[MemoryItem]) -> list[str]`. Stores
  one or more memory units for the user and returns backend ids. Each
  `MemoryItem` carries a `content` string plus optional `producer`, `role`,
  `timestamp`, and `metadata`. The adapter decides how a batch of items maps
  to backend requests (it may batch them). One `add` op records the number of
  ids returned as `n_items`.
- **SEARCH** — `client.search(user, query: QueryItem) -> list[ResultItem]`.
  Retrieves memories scoped to this user only. The `QueryItem` carries a
  `query` string, a `top_k` (default 20, set via `--top-k`), and two optional
  server-side knobs: `expand_context` (`--expand`, neighbouring episodes around
  each hit) and `filter` (`--filter`, a metadata filter expression such as
  `metadata.category=cat_3`). Both default to inert — the adapter omits them
  from the wire payload when unset, so a default run's request is unchanged.
  One `search` op records the number of results returned as `n_items`.

Per-request recording: `op_type`, `user_id`, `started_at`, `ended_at`,
`status` (ok / error / rejected), `error_kind`, `n_items`. The aggregate
also reports per op `items.empty_rate` — the fraction of *successful* ops
that moved nothing — which for search is the only way to tell a run where
every query returned zero results from a healthy one (both have a 0% error
rate).

### What data add and search operate on

Neither the scenarios nor the runner decide *what content* is added or what
*query* is searched — that comes entirely from the **dataset adapter**.
A scenario only decides *when* and *how often*; the data is fixed by the
dataset (and the seed for replication). This keeps the load shape separate
from the data shape.

**`add` data — `dataset.memory_stream(user)`** yields `MemoryItem`s, one per
stored unit. The adapter owns the content; the backend adapter forwards
`MemoryItem` fields to the wire. Per dataset:

- **LongMemEval** — a sample's `haystack_sessions` are flattened into turn
  `content`s, each split into <=3000-char chunks on word boundaries. One
  turn can become several `MemoryItem`s. `role`/`timestamp`/`metadata` are
  not set; only `content` and `producer` (the `UserId`) are sent. A user's
  stream is its sample's entire haystack (typically tens to hundreds of
  chunks).
- **Synthetic** — `memories_per_user` (default 100) deterministic items of
  `content_chars` (default 200) each, generated from the seed. An optional
  `categories` option writes `metadata.category` (`cat_<i mod N>`) so a
  single-value `--filter` selects about `1/N` of the data; with it unset the
  corpus is byte-identical to one built without it. Reproducible and
  download-free.

Every `MemoryItem` is sent as an **episodic** memory (`types: ["episodic"]`,
currently hardcoded; semantic is a future option). In practice each `add` op
carries one item and becomes one request.

**`search` data — content-derived queries.** Search queries are **not**
taken from a dataset evaluation question. `search-load` and `mixed`
build the query pool from the user's own `memory_stream` items (one
`QueryItem` per stored unit, `query` = the item's content, `top_k` from
`--top-k`, default 20, plus the optional `expand_context`/`filter` from
`--expand`/`--filter`). The
pool is therefore as large as the memory stream, so cycling it does not
naively repeat a single query — important because a tiny, fixed query pool
would warm a server's result cache and understate search latency. The pool
is cycled with a rotating per-pass start offset and a small **think jitter**
(`delay`) so users drift out of lockstep. LTM100 does not measure recall
quality; gold `expected` fields, if a dataset carries any, are not scored.

> **`chat-replay` derives queries from the conversation.** Instead of the
> memory pool, `chat-replay` uses the dataset's optional
> **`turn_stream(user)`**, which yields `(role, items)` turns in conversation
> order. The recall query before a user turn is that turn's first item's
> content — recall driven by the conversation itself, exactly as a live
> chatbot queries its memory with the user's utterance. LongMemEval exposes
> `turn_stream` from its `haystack_sessions` (user/assistant turns, each
> chunked the same way as `memory_stream`); synthetic does not, and a
> `chat-replay` run against a dataset without it fails loudly at validation
> time.

### Wrap-around

Every scenario's plan is **infinite**: it wraps its underlying stream
(`memory_stream`, the query pool, or `turn_stream`) and keeps emitting until
the runner stops it. This matters for `--duration` runs — a finite plan
would exhaust a short stream and leave the rest of the duration idle. The
runner bounds consumption instead (count- or time-based for closed;
`--session-ops` per arriving session for open). Count-based termination is
exact: exactly `--ops` results are recorded.

### Termination

Every run terminates by **either** `--duration SECONDS` or `--ops N`
(exactly one is required). The open model additionally requires `--duration`.

### Axis separation: load model vs. scenario

The **load model** (closed/open) and the **scenario** are independent axes.
A scenario owns the op mix and the data it emits; the runner owns the
consume schedule. The open model consumes a bounded slice of the same
`Scenario.plan()` the closed model loops over. **Every scenario runs under
both `--model closed` and `--model open`**:

- **closed**: a fixed pool of `--users N` each looping the plan back-to-back
  (with think jitter), in-flight = 1 per user, optionally capped by
  `--global-concurrency`.
- **open**: a Poisson arrival process spawns sessions at `--arrival-rate`;
  each session draws a user round-robin and consumes up to `--session-ops`
  ops of the plan, under the global-cap + `--queue-bound` rejection policy.

`chat-replay` under closed is a deterministic, in-order replay; under open it
is the most realistic chatbot load (Poisson-arriving sessions, each replaying
a slice of the conversation with the congestion policy in effect). The
recall cadence and turn content are identical in both.

For the load-model mechanics in depth (the closed loop vs the arrival
process, the congestion/rejection policy, and when to use each model), see
[`load-models.md`](./load-models.md).

### Op mix ownership

The op mix (add vs search) is owned by the scenario for **both** load models.
There is no runner-level mix weight. In the open model, arriving sessions
consume ops from the same `Scenario.plan()` interface the closed model loops
over; the runner only decides *how many* ops each session takes.

---

## `chat-replay` — the primary workload

**Tests:** a real chatbot-with-LTM integration workload — recall before
answering, then ingest the conversation turn, replayed over a multi-turn
dialogue. Closest to how an LTM is actually used in production, and the
primary workload of LTM100.

**User behavior:** the user walks its dataset's structured `turn_stream`
(user/assistant turns in conversation order), which wraps so a duration run
replays the conversation as many times as needed. For every turn:

- if it is a **user** turn: issue a SEARCH whose query is the user turn's
  content when this turn's recall cadence fires (see `search_every`), then
  ADD the turn's items;
- otherwise (assistant turn): just ADD the turn's items.

So a user/assistant turn-pair becomes `search → add (user) → add
(assistant)` when recall fires. Recall is driven by the upcoming user turn's
content — exactly as a live chatbot queries its memory with the user's
utterance. Adds and the recall search are interleaved as a real session
interleaves them.

**Users:** `--users N`.

**Concurrency:** per-user in-flight 1; the conversation is replayed in order.

**add:** one item per chunk of each turn's content, `delay = uniform(0,
think)` (default 0.05). Both user and assistant turns are added identically
(episodic, `producer` = user id).
**search:** one per recall-firing user turn, `query` = that turn's first
chunk's content, `top_k` from `--top-k` (default 20), plus the optional
`expand_context`/`filter` from `--expand`/`--filter`, `delay = uniform(0, think)`.
The query pool is **not** used — queries come from the turn stream.

**Parameters:** `--think` (default 0.05), `--search-every N` (default 1 =
recall before every user turn; N>1 recalls only every Nth user turn), `--top-k`
(default 20, recall search depth). The user-turn counter resets each replay
pass, so each pass is an independent, reproducible chat session with the same
recall pattern.

**LLM answer time and user think time** (`--answer-time`, `--user-gap`, both
default 0 = back-to-back): a real chatbot does not loop back-to-back — after
recalling, the LLM spends time generating an answer, and the user spends
time reading/typing before the next turn. These are modeled as two *mean*
delays (Exponential, the same distribution the open-model arrival process
uses):

- `--answer-time T`: a delay ~Exp(mean=T) is attached to the **last ADD** of
  each user turn — the assistant turn's adds (the LLM writing its answer)
  happen during this gap. Models LLM answer-generation time.
- `--user-gap T`: a delay ~Exp(mean=T) is attached to the **first** op (the
  recall SEARCH, or the first ADD if recall is skipped via `search_every`)
  of a user turn, except the very first user turn of each replay pass (so
  each pass starts cleanly). Models the user reading the prior reply and
  typing the next utterance.

Both apply **uniformly to all users** — the same mean for every user. (A
per-user ratio for finer control is a planned follow-up.) With both at 0,
chat-replay reproduces the original tight `search → add → search → add` loop;
raising them spreads the load out, lowering concurrency toward a realistic
chatbot session shape.

**Dataset requirement:** the dataset must expose `turn_stream` (LongMemEval
does; synthetic does not). The runner validates this before the run and
raises loudly if it is missing — a chat-replay run against a dataset without
dialogue structure fails immediately rather than silently.

**Pre-ingest:** not needed — the user adds its own conversation as it goes
and recalls against what it has stored so far.

**Termination:** `--duration` or `--ops`. The turn stream wraps, so the user
keeps replaying until the runner stops it.

**Load model:** runs under both `--model closed` (a deterministic, in-order
replay) and `--model open` (Poisson-arriving sessions, each replaying a
slice of the conversation under the congestion policy — the most realistic
chatbot load).

---

## `add-load`

**Tests:** pure storage (ingest) throughput — how fast the backend stores
memories.

**User behavior:** each user iterates its `memory_stream`, emitting one
`Op(ADD, items=[item])` per item, back to back. The stream wraps, so a
duration run sustains add load until the runner stops it. No search is ever
issued.

**Users:** `--users N`, N coroutines launched together.

**Concurrency:** per-user in-flight 1; N users add in parallel, so
simultaneous `add` calls = `min(N, C)`. With no global cap, N concurrent adds.

**add:** back-to-back, `delay=0`. One op per memory item. Synthetic yields
`memories_per_user` (default 100) items; LongMemEval yields one add per
haystack chunk.

**search:** none.

**Termination:** `--duration` or `--ops`.

---

## `search-load`

**Tests:** pure search throughput and latency — read-path load against
pre-populated memory.

**User behavior:** each user loops a content-derived query pool (built from
its own `memory_stream`), emitting SEARCH ops forever. No `add` during the
measured run, so **memory must already be present** (use `--preingest`).

**Users:** `--users N`.

**Concurrency:** per-user in-flight 1; the pool wraps, so only
`--duration`/`--ops` terminates the run.

**search:** each query carries a small think time
(`delay = uniform(0, 0.02)`) so users drift out of lockstep. The query pool
is the user's own memory contents, cycled with a rotating per-pass start
offset so passes are not identical. `top_k` comes from the `QueryItem`
(set via `--top-k`, default 20); `expand_context`/`filter` come from
`--expand`/`--filter` when set.

**add:** none during measurement.

**Pre-ingest:** essential here. `--preingest` fills each user's memories
before the measured run, under the global concurrency cap, ingesting a
`--preingest-fraction` (default 1.0 = all) of each user's `memory_stream`.
Pre-ingest is excluded from metrics.

**Termination:** `--duration` or `--ops`.

---

## `mixed`

**Tests:** a controllable add/search mixture — a flat op stream with a
tunable search/add ratio that works with any dataset (including the
synthetic one, which has no `turn_stream`). Handy as a quick congestion
probe under either load model: it exercises emergent concurrency and the
congestion/rejection policy under overload without needing a dialogue.

This is the lightweight, no-dialogue workload: unlike `chat-replay` it does
not replay a conversation, so it has no `turn_stream` requirement and runs
equally against synthetic data.

**User behavior — two-stage (open model):**

1. **Arrival process (runner-owned, not scenario).** A Poisson process spawns
   sessions at `--arrival-rate λ`. Inter-arrival is `expovariate(λ)`. Each
   arrival draws the next user round-robin from the `--users N` pool (so the
   tenant already exists) and starts one **session**. Sessions keep arriving
   until the deadline.

2. **Session (consumes the scenario plan).** An arriving user consumes up to
   `--session-ops` ops from `self.scenario.plan(...)`, then leaves. The op
   mix is decided by the scenario:
   - `rng.random() < search_weight` (default 0.8) → SEARCH (query drawn from
     the content-derived pool, the user's own memory contents)
   - else → ADD (one item from `memory_stream`, cycled when exhausted)
   - each op carries think jitter (`delay = uniform(0, think)`, `think`
     default 0.05)

Under the **closed** model, the same plan is simply looped back-to-back by a
fixed pool of N users, so the op mix still applies and `--duration`/`--ops`
bounds the run.

**Users:** `--users N` is the **tenant identity pool, not the concurrent
count** (under open). Concurrency is emergent — a function of arrival rate
vs service rate. The same user may appear in multiple concurrent sessions
(same tenant, isolation preserved). Under closed, N is the fixed concurrent
count.

**Concurrency (congestion policy — the core output under open):**
- `--global-concurrency C`: max in-flight.
- `--queue-bound Q`: how many requests beyond C may wait in queue.
- When in-flight reaches `C + Q`, the next request is **rejected**
  (`status=rejected`, `error_kind=queue_full`, zero latency) — it is recorded
  but not executed.
- `Q=0` rejects immediately on C saturation. `Q>0` lets up to Q requests
  queue (busy-wait in 0.005s steps) before acquiring a slot.

**add:** when not a search, one item per op, `delay = uniform(0, think)`.
**search:** query from the content-derived pool (the user's own memory
contents), cycled, `top_k` from `--top-k` (default 20), plus
`expand_context`/`filter` from `--expand`/`--filter` when set,
`delay = uniform(0, think)`.

**Parameters:** `--search-weight` (default 0.8, forwarded to the scenario
constructor), `--think` (default 0.05). The open-model knobs `--arrival-rate`,
`--session-ops`, `--queue-bound` live on `RunConfig`.

**Pre-ingest:** recommended — arriving/looping users search against memory
that should already exist.

**Termination:** `--duration` or `--ops` (closed); `--duration` is required
under open (sessions arrive until the deadline, then in-flight sessions
drain).

---

## Comparison

| | chat-replay | add-load | search-load | mixed |
|---|---|---|---|---|
| load model | closed + open | closed + open | closed + open | closed + open |
| ops | recall + add per turn | add only | search only | search-weighted add + search |
| search query source | user turn content | — | own memory content | own memory content |
| user lifetime | wraps the dialogue | wraps to sustain | wraps to sustain | per-session (arrival → `session_ops`) |
| concurrency | N fixed, in-order replay | N fixed, parallel add | N fixed, parallel search | emergent under open; N fixed under closed |
| precondition | dataset with `turn_stream` | none | preingest required | preingest recommended |
| termination | duration/ops | duration/ops | duration/ops | duration required (open) |
| op scheduling | think 0–0.05s | back-to-back | think 0–0.02s | think 0–0.05s |
| key output | chatbot-LTM integration load | write throughput | read latency | rejection/congestion metrics (open) |

---

## MemMachine backend specifics

The above describes behavior at the `LTMClient` contract level. When the
backend is **MemMachine over REST** (the current baseline adapter,
`ltm100/adapters/backends/memmachine.py`), the contract maps to concrete
requests as follows.

**Tenancy.** MemMachine's multi-tenancy is
`session_key = f"{org_id}/{project_id}"`. The adapter maps a user to a single
org and one project per user:

```
session_key = f"{org_prefix}/user_{UserId}"
```

So **one user = one MemMachine project** under a shared org. `setup` creates
one project per user (409 "already exists" is tolerated for reruns);
`teardown` deletes them.

**Isolation scope.** Two backend options change that shape, so a run can
measure the cost of the boundary itself:

```yaml
backend:
  project_id: shared          # every user lands in this one project
  filter_by_producer: true    # AND producer_id = '<user>' into every search
```

`project_id` collapses every virtual user onto one project; `filter_by_producer`
then makes `producer_id` do the separation the project boundary used to do. It
is AND-ed into any `--filter` you pass rather than replacing it, so an arm can
carry a metadata filter at the same time and the two stay separable. Your filter
is parenthesised when the two are combined, because `AND` binds tighter than
`OR` server-side: `producer_id = 'u' AND a OR b` parses as
`(producer_id = 'u' AND a) OR b`, which returns anything matching `b` whoever
produced it. Both are
inert unset: `project_id: ""` keeps one project per user and the request is
byte-identical to one built without either option. `filter_by_producer` also
works without `project_id`: in a per-user project every episode's producer is
that user, so the results do not change and the arm measures what the filter
alone costs — the natural control for a shared-project arm.

One restriction: each shard tears down the users it drove, which assumes a
project per user. With `project_id` set they all share one, so `--procs > 1`
refuses to delete on exit rather than let the first shard to finish drop the
project the others are still using. Pass `--no-delete-on-exit` — which an
isolation-scope arm wants anyway, since the corpus is the thing under test. For
the same reason, a shared project that already existed when the run started is
never deleted on exit; only one the run created is.

This is a different measurement, not a variant of the same one. MemMachine
partitions its vector collection by a key derived from `org_id/project_id` and
builds it with `m=0, payload_m=16`: there are no global HNSW links, only the
per-value links Qdrant adds for each indexed field — the partition key among
them — to each segment's one graph. A partition below the full-scan threshold is
searched exactly rather than through those links. So many per-user projects and
one shared project take different search paths, and the partition filter
decides which one a query gets.

Read such an arm carefully: collapsing users moves three things at once — the
vector index topology, the segment-store partitioning (one partition instead of
N, which is database-side rather than vector-side), and query isolation, since
`top_k` now draws from every user's corpus. Run the matched per-user control and
split the server's own `event_memory_query_phase_seconds` by phase — the core
exposes it on `GET /api/v2/metrics`, alongside
`vector_store_qdrant_latency_seconds` and
`segment_store_sqlalchemy_latency_seconds`, so reading the delta across a run
separates vector-side from database-side. Without that split the arm cannot say
which of the three moved.

Two more things worth knowing before scoring one. `producer_id` is one of ten
filterable server-side fields, all of which have a vector-store index behind
them; user metadata under the `m.` prefix does not, so the two are not
comparable as filters.

**Check the filter contract before trusting a filtered number.**
`tools/filter_contract.py` ingests a small fixture into one project and asserts
what the server does with a filter — that a producer filter returns exactly one
producer, that a restrictive filter returns the match count rather than a padded
`top_k`, that an `OR` cannot widen past the producer scope, and that an unknown
field is rejected rather than ignored:

```sh
./tools/filter_contract.py http://<core-pod-ip>:8081
```

Twelve checks, a few seconds, no load. It exists because the unit suite covers
only the filter string the client builds, and every filter defect found so far
was on the other side of the wire while those tests stayed green. Run it
whenever the build, the vector store or the grammar changes.

The embedding call usually dominates a single search, so end-to-end latency
cannot resolve a change in the vector store. Score on the phase metrics, or use
a local embedder.

**add** maps to `POST /api/v2/memories`:

```
org_id, project_id              # from the user's session_key
types: ["episodic"]             # episodic only (hardcoded; semantic is a future option)
messages: [{ content, producer, role?, timestamp?, metadata? }]
```

`MemoryItem.content` is passed through; `producer` is the `UserId`; `role`,
`timestamp`, `metadata` are forwarded if present (metadata values are
stringified). The adapter chunks `items` into batches of `add_batch_size`
(YAML, default 50) per request. Most scenarios pass one item per op, so one
op typically becomes one request.

**search** maps to `POST /api/v2/memories/search`:

```
org_id, project_id
query: <string>
top_k: 20                      # from the QueryItem (--top-k, default 20)
types: ["episodic"]
expand_context?: <int>          # only when --expand is set (else omitted)
filter?: <expr>                # only when --filter is set (else omitted)
```

`expand_context` and `filter` are omitted from the payload when unset, so a
default run's request is byte-identical to one built without them. The
response's `content.episodic_memory.long_term_memory.episodes` is parsed
into `ResultItem`s (`content`, `score`, `uid`, `metadata`).

**Retries.** The REST transport retries only *connection-level* failures
(timeout, connection error) up to the backend's `retries` option (default 0),
with exponential backoff. An HTTP error status is a real answer from the
server and is never retried — retrying it would understate the error rate.

**Endpoints used:**

```
POST /api/v2/projects          create a per-user project (setup)
POST /api/v2/projects/delete   delete a project (teardown)
POST /api/v2/memories          add memories
POST /api/v2/memories/search   search memories
GET  /api/v2/health            readiness check + build/version probe (meta.build)
```

> Note: MemMachine's `projects/list` is eventually consistent — an immediate
> list after delete may still show a project before it disappears. The
> delete itself is confirmed by the server's response, not by listing.

### MemMachine-MCP transport

The `memmachine-mcp` backend adapter reuses this exact contract under the same
`LTMClient` interface, so every scenario runs identically; only the wire path
differs. The measured `add`/`search` ops call MemMachine's MCP tools
(`add_memory` / `search_memory`) mounted at `/mcp` on the same server, via
`fastmcp.Client`. Tenancy is passed as tool arguments (`org_id` / `proj_id` /
`user_id`), mapped the same way as REST.

Lifecycle is hybrid: `setup`/`teardown` (project create/delete) still go
through the REST endpoints above, because the MCP server exposes no
project-management tools. Provisioning is out of measurement, so mixing
transports there does not affect the measured add/search path.

Two differences versus the REST adapter, by design and worth noting when
comparing the two transports:

- `add_memory` writes `types=ALL_MEMORY_TYPES` (episodic **and** semantic).
  The REST adapter is episodic-only. Semantic memory triggers LLM-based
  background processing, so MCP `add` latency is not directly comparable to
  REST `add` latency.
- `add_memory` returns a success status with no ids, so `add` reports
  `n_items` as the number of items sent (one `add_memory` call per item),
  whereas REST counts returned uids.
- `search_memory` returns a `SearchResult` with the same
  `content.episodic_memory.long_term_memory.episodes` shape the REST search
  endpoint uses, so results parse identically.

**Knobs the MCP tools cannot honour.** `add_memory` has no metadata field,
and `search_memory` exposes neither `expand_context` nor `filter`. Rather
than silently dropping metadata or running a baseline search under the label
of a filtered/expanded one (which would make the error rate lie), the MCP
adapter **raises** for `--expand`/`--filter` and for items carrying metadata
— use the REST backend for those arms. The same goes for the isolation-scope
options: `project_id` and `filter_by_producer` are refused at construction.

---

## Mem0 backend specifics

The `mem0` adapter targets the unversioned self-hosted Mem0 OSS REST API.
Each virtual user maps to `f"{user_prefix}_user_{UserId}"` and is passed as
Mem0's `user_id` on every operation. Mem0 creates that scope lazily, so
`setup` makes no request.

**add** sends one `POST /memories` request per `MemoryItem` with `messages`,
the scoped `user_id`, optional metadata, and `infer`. `infer` defaults to
false, preserving one-input/one-memory accounting and avoiding LLM fact
extraction in comparisons with MemMachine's episodic-only path. With
`infer: true`, one input may produce zero, one, or several memories and add
latency includes the extraction pipeline.

**search** sends `POST /search` with `query`, `top_k`, and a `filters` object
containing the scoped `user_id`. LTM100's exact-match
`metadata.key=value` expression is translated to Mem0's flattened metadata
filter `{key: value}`. Mem0 has no `expand_context` equivalent, so non-zero
`--expand` is rejected instead of silently ignored.

**teardown** sends `DELETE /memories?user_id=...` once per virtual user when
delete-on-exit is enabled. Optional `api_key` authentication is sent through
the `X-API-Key` header.
