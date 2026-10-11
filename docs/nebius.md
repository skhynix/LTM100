# Nebius agent trajectories

The `nebius` dataset adapter reads
[SWE-rebench OpenHands trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories).
The source contains recorded attempts by Qwen3-Coder-480B-A35B-Instruct with
OpenHands v0.54.0 to solve GitHub issues. It is not a history of identifiable
human developers and does not record LTM add/search calls.

## Current scope

The adapter provides two finite, repeatable views of each user's assigned data:

- `task_stream(user)`: structured `AgentTask` records retaining source message
  order, roles, tool arguments, call/result IDs, terminal response, and source
  outcome. No tool commands are executed and no missing results are generated.
- `memory_stream(user)`: a text/chunk projection for existing `add-load`,
  `search-load`, and `mixed` scenarios. This is a corpus workload, not an
  agent's actual memory-use policy. Queries remain content-derived by the
  existing scenarios.

`chat-replay` is rejected through its existing dialogue capability check:
agent trajectories do not expose `turn_stream` or `session_stream`. The
[coding-agent scenario](coding-agent.md) consumes `task_stream` and defines
hook/MCP policies, task repetition, and simulated agent/tool timing separately.

## Configuration and loading

Start with [`examples/nebius.yaml`](../examples/nebius.yaml). Scenario, user
count, seed, and termination remain CLI options, as with LongMemEval.

| Dataset option | Default | Meaning |
|---|---|---|
| `name` | Required: `nebius` | Adapter registry name |
| `length` | All rows | First N source trajectories; non-negative integer |
| `revision` | `35455389ab51bf5e2306bfd436ef72d0f98bf882` | Pinned Hugging Face dataset revision |
| `cache_dir` | Hugging Face default | Download cache for the source Parquet |
| `path` | Hugging Face source | Local `.parquet`, `.json`, or `.jsonl`; takes precedence |
| `chunk_chars` | 3000 | Maximum characters in each projected memory chunk |

Install `pip install -e ".[datasets]"` for Hugging Face/Parquet loading. Local
JSON/JSONL uses the normal runtime dependencies. JSON must be an array of
source-shaped records; JSONL must contain one record per non-blank line.

Hugging Face loading downloads the pinned `trajectories.parquet` once and
reuses its cache. `length` limits selected records, not download bytes. Parquet
loading projects the required columns and caches one decoded row group,
without expanding the file into a second full Arrow dataset. Row-group memory
depends on how the source Parquet was written. JSON/JSONL retains the selected
raw records in memory.

Before backend setup, `users()` validates all selected trajectories and rejects
duplicate trajectory IDs or malformed structure with source row context.
Successful validation is retained for subsequent user initialization; complete
normalized tasks are not retained. Loader/cache/auth failures propagate rather
than silently selecting another source or dropping invalid records.

Existing generic scenarios materialize per-user memory/query pools. Choose a
manageable `length` for these workloads: the full trajectory corpus is much
larger than its compressed Parquet. Random access across many row groups can
also be expensive. `chunk_chars` counts characters, not bytes or tokens.

## User mapping and repetition

The adapter shuffles selected row indices with the CLI seed, then distributes
them round-robin across the whole run's user population. With M trajectories
and N users:

- M >= N: every trajectory is assigned once; user task counts differ by at
  most one. Different trajectories can still describe the same GitHub issue.
- M < N: users receive one trajectory each through cyclic replication. Backend
  user identities and producer fields remain independent.
- Every worker builds the same global assignment before user sharding;
  `--procs` does not change source task lists or corpus payloads.

For a shuffled A-H pool with three users, task lists are A/D/G, B/E/H, and C/F.
`task_indices(user)` exposes selected-source row indices for inspection.
Repeated calls to either stream reproduce the same finite source view. A new
`users()` call replaces the assignment; consumers should obtain streams after
initializing the run's users.

The adapter does not loop indefinitely. Existing generic scenarios own their
usual item/query wrap-around. Agent task cycling and task gaps belong
to the coding-agent scenario; they must preserve each trajectory boundary.
Repeated adds are new requests, and retained storage depends on the backend's
duplicate handling. Repetition never implies a memory reset.

Changing user count changes each user's task list. Assignment is synthetic:
tasks can come from different repositories and are not claimed to be a real
developer's longitudinal history.

## Projection and source semantics

System prompts and global tool definitions are excluded from the generic
corpus. User/assistant/tool messages retain their source role on each chunk,
and producer is the LTM100 virtual user. Backend role support remains subject
to the [support matrix](backend-support.md); no dataset-specific metadata is
required.

Assistant text is combined with tool names and deterministically serialized
arguments. This preserves actions even when assistant content is empty. Tool
observations carry a tool-name prefix. A terminal `finish.message` is included
once alongside any distinct assistant text, without storing the finish wrapper
again. Chunking does not modify the structured source messages.

`exit_status: submit` means the source agent terminated; it does not mean the
issue was solved. `resolved` is source task metadata, never an LTM error rate
or a retrieval-quality score. Interrupted attempts retain their recorded
actions, including an unobserved final call, without an invented final answer.

Some source `user` messages are framework feedback, such as action-processing
errors or requests to continue working. Preserve them without assuming they
are new human prompts. Source roles alone do not define chatbot timing.
The dataset has no consistent measured message/tool timestamps, so agent
timing is modeled by the coding-agent profile, not inferred from the source.

## Source validation and attribution

The pinned revision was inspected on 2026-10-10: all 67,074 trajectories
normalized successfully, with 6,306 distinct issue IDs and 1,823 repositories.
There were 286 trajectories with multiple source `user` messages. This checks
record structure, not correctness of source code patches or memory retrieval.

The dataset is licensed under **CC BY 4.0**, independently of LTM100's Apache
2.0 code license. Dataset files are not bundled in this repository. Attribute
the source when redistributing its contents; review underlying code licenses
for separate code reuse. See the
[dataset card and citation](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
and [Nebius report](https://nebius.com/blog/posts/openhands-trajectories-with-qwen3-coder-480b).
