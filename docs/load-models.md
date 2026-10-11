# Load models

LTM100 separates the **scenario** (what ops to emit and on what data) from
the **load model** (when and how many users emit them). The same scenario
plan runs under either load model — only the consume schedule changes. This
document explains the two models, the congestion policy, multi-process
load generation, and when to use each. For per-scenario data flow, see
[`scenarios.md`](./scenarios.md).

## The two axes

- **Scenario** — owns the *op mix* (add vs search) and the *data* (which
  memory content to add, which query to search). Its `plan()` yields `Op`
  requests and optional `Wait` timeline events. Plans may wrap indefinitely
  or stop at a source boundary; the scenario owns that choice.
- **Load model** — owns the *consume schedule*: who consumes the plan, when,
  and for how long. The runner drives this; it never touches the op mix or
  data.

Because the axes are separate, every scenario runs under both `--model
closed` and `--model open`.

## Closed model (`--model closed`, default)

A fixed pool of `--users N` virtual users, each an asyncio coroutine that
loops the scenario plan back-to-back:

- **Per-session in-flight = 1.** A session lane sends one request, *waits for
  the response*, then sends the next. Ordinary workloads use one lane per
  user; profiled `chat-replay` can create several lanes sharing one user.
- **Concurrency = min(session lanes, `--global-concurrency`)** (all lanes if
  no cap). Because each lane blocks on its response, service time feeds back
  into concurrency (a *closed* loop).
- **`N` is the concurrent user count.** `--users 20` models "20 users always
  connected".
- **Timing** includes per-request think time (`Op.delay`) and optional
  request-free `Wait` events. `add-load` is back-to-back; chat-replay models
  answer/user gaps, while coding-agent models agent/tool/task time. These
  waits do not occupy request slots or enter request latency/QPS.

Closed models the classic load test: "with N users hammering the server,
how fast is it?" Use it to measure raw throughput and latency at a fixed
concurrency.

## Open model (`--model open`)

Sessions *arrive* over time, run a bounded slice, then leave:

- **Arrival process (runner-owned).** A Poisson process spawns sessions at
  `--arrival-rate λ`. Inter-arrival is `expovariate(λ)` — bursty but with a
  fixed average rate. Sessions keep arriving until `--duration`.
- **Session lifetime is bounded.** Each arriving session consumes up to
  `--session-ops` requests from the scenario plan, then the session ends. A
  finite plan may finish earlier. Coding-agent consumes one assigned task
  per admission and never crosses into the next task; `Wait` events do not
  consume this request budget.
- **Arrival is the input; concurrency is the output.** `λ` fixes how fast
  sessions arrive; *how many run at once* is whatever the server's service
  rate produces. A fast server keeps concurrency low; a slow one lets it
  pile up. This is an *open* loop — arrival is independent of service rate.
- **Tenancy pool.** Arriving sessions draw users round-robin from the
  `--users N` pool (so the tenant already exists). The same user may appear
  in several concurrent sessions (same tenant; isolation is preserved).
- **`--duration` is required** (open runs are always time-bounded).

Open models real traffic: visitors arriving independently (web site,
messaging app) and each doing a short burst of work.

## Termination and request draining

`--duration` bounds when the runner may dispatch new backend operations. Delays,
ramp-up, and waits for a global concurrency slot stop at the phase deadline.
An operation that expires before dispatch is omitted from results; it is not
an error or a `queue_full` rejection. Open-model session admissions also stop
at the deadline.

Requests already dispatched are allowed to finish and their results are
recorded. The measured window, final server-metrics snapshot, and time-series
reports include this drain, so a run's completion time can exceed its duration.
The boundary is client dispatch, not server arrival or completion; transport
retries remain part of the already-dispatched operation.

A closed run with only `--ops N` executes exactly N operations, provided the
scenario supplies enough operations. Count exhaustion never discards another
lane's reserved work. If both `--ops` and `--duration` are set, duration also
limits dispatch and fewer than N operations may execute. Open runs remain
duration-based, with `--session-ops` limiting each session.

Warm-up applies the same deadline and drain rules, then resets termination
state before the separate measured duration or operation budget begins.

Coding-agent with `task_repeat: false` can exhaust its assigned task pool
before the deadline. It permits duration-plus-ops core configurations, but
rejects pure count-based closed runs because the exact budget may be
unreachable. See [coding-agent replay](coding-agent.md).

If open admission or source preparation fails, the runner stops admissions
and further request dispatch, returns an unowned session-lane reservation,
and drains existing session tasks before propagating the original error.
Already executing requests may finish; delayed or queued requests do not
start after that failure. Pending waits remain bounded by the phase deadline.
The caller can then safely perform teardown and close the transport.

## Congestion policy (the open model's key output)

Under closed, concurrency is fixed at N so nothing is ever rejected. Under
open, if arrivals outrun the server, requests pile up — and LTM100 must
decide what to do. The policy:

- `--global-concurrency C`: max requests in flight at once.
- `--queue-bound Q`: how many requests beyond `C` may wait for a slot.
- When in-flight reaches `C + Q`, the next request is **rejected**:
  - `status = "rejected"`, `error_kind = "queue_full"`, zero latency
    (it is recorded but not executed, so the rejection rate is measurable).
- `Q=0` rejects immediately once `C` is saturated (no queue). `Q>0` lets up
  to exactly `Q` requests wait for a slot before rejecting. Admission is an
  atomic reservation over running plus waiting requests, so simultaneous
  arrivals cannot oversubscribe the queue.

The rejection rate and where it kicks in are the open model's most
important result: "at what arrival rate does the server start dropping
load?" Find it by sweeping `--arrival-rate` upward and watching the
rejected share.

Reports keep the overload populations separate: `offered` is every attempt,
`accepted` is every request admitted to service (`successful + errors`), and
`rejected` is a queue-full refusal. `throughput_ops_s` and `qps` report
successful throughput; explicit `*_ops_s` fields expose offered, accepted,
successful, and rejected rates. Rejection rate is rejected / offered, while
error rate is backend errors / accepted. Service-latency percentiles contain
successful requests only, so zero-time rejections cannot lower p50 or p99.

### Profiled chat-session admission

`chat-replay` profiles add an earlier, session-level admission layer through
`max_sessions_per_user`, which is required for every group in a profiled open
run. Each Poisson arrival is first assigned to its intended user round-robin.
If that user already has the configured number of active chat sessions, the
session is rejected; it is never moved to another tenant, and `--arrival-rate`
is not multiplied. An admitted session occupies one reusable
source-conversation lane until it finishes its `--session-ops` slice.

Session admission appears under `summary.sessions` overall and by workload
group. It is intentionally separate from request rejection: admitted sessions'
individual operations still pass through `--global-concurrency` and
`--queue-bound`.

## Same scenario, two models — chat-replay example

| | `chat-replay --model closed` | `chat-replay --model open` |
| --- | --- | --- |
| users | N always connected | sessions drawn from the pool, then leave |
| per-user lifetime | forever (dialogue wraps) | `--session-ops` ops per session |
| concurrency | configured session lanes | emergent (arrival vs service rate and optional per-user caps) |
| timing | think time only | Poisson arrival + think |
| rejections | none | session-admission and `queue_full` rejections possible |
| models reality | "20 chat windows kept open" | "customers starting support chats at random" |

Both consume the *same* `plan()` — recall order, add content, and
`--search-every` cadence are identical. Only who fires when differs.

## When to use which

- **Closed** — measure peak throughput/latency at a fixed concurrency; the
  standard load-test shape. "How fast is the server when N users never stop?"
- **Open** — observe behavior under realistic arrival patterns and find the
  overload/rejection threshold. Sweep `--arrival-rate` up and watch
  `queue_full` rejections climb — the point where they start is the
  server's effective capacity.
- **Most realistic chatbot load** — `chat-replay --model open`: customers
  arrive per a Poisson process, each session recalls then answers/ingests a
  slice of the conversation, under the congestion policy.

## Multi-process load generation (`--procs`)

One asyncio event loop saturates a single CPU core. If the server answers
fast enough, the client becomes the slower side and the run starts
describing the generator rather than the target. `--procs N` fixes that by
running the same `LoadRunner` in N OS processes, each driving a disjoint
slice of the virtual users.

- **Sharding.** Users are partitioned round-robin across the shards, so an
  ordered dataset does not hand one shard all the large conversations. Each
  shard builds its own runner, so whole-run budgets are divided across them:
  `--ops`, `--global-concurrency`, `--queue-bound` are split as integers
  (the remainder goes to the lowest-numbered shards, so shares sum exactly),
  and `--arrival-rate` is divided by N (the combined rate is preserved).
  `--session-ops` is per-session and `--users` is partitioned by the shard,
  so neither is divided.
- **Pooling.** Shards return their raw `OpResult`s, not summaries; the
  parent pools them and runs the same `aggregate` used for a single-process
  run, so percentiles are computed over the whole population — no
  approximation from per-shard percentiles.
- **Spawned, not forked.** Workers are spawned (a forked child inherits the
  parent's event loop and open sockets, which asyncio does not support).
- **Reproducible.** The per-user seed is derived from the run seed and the
  user id, not the shard index, so the same run shape reproduces regardless
  of `--procs`.
- `--procs 1` is the original single-process topology, exactly the same
  code path.

Use `--procs` when the server is fast and you suspect the client is the
bottleneck (a single-process run whose throughput stops climbing as you add
users is the tell). The two load models above are unchanged by sharding — a
shard simply drives fewer users.
