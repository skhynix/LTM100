# Example configurations

Scenario selection is a CLI option, while each YAML selects a dataset and a
backend. The dataset determines whether `chat-replay` is available.

| Config | Dataset | Backend | Recommended scenarios |
|---|---|---|---|
| `memmachine.yaml` | LongMemEval | MemMachine REST | `chat-replay` |
| `memmachine-memory-types.yaml` | Synthetic | MemMachine REST (semantic opt-in) | `add-load`, `search-load`, `mixed` |
| `synthetic.yaml` | Synthetic | MemMachine REST | `add-load`, `search-load`, `mixed` |
| `nebius.yaml` | Nebius agent trajectory corpus | MemMachine REST | `add-load`, `search-load`, `mixed` |
| `memmachine-mcp.yaml` | LongMemEval | MemMachine MCP | `chat-replay` |
| `mem0.yaml` | Synthetic | Mem0 REST | `add-load`, `search-load`, `mixed` |
| `supermemory.yaml` | Synthetic | Supermemory REST (direct memories) | `add-load`, `search-load`, `mixed` |
| `shared-project.yaml` | Synthetic | MemMachine REST | Isolation-scope `add-load`, `search-load`, or `mixed` |
| `chat-profile.yaml` | Chat workload profile | Used with a LongMemEval config | Group-specific `chat-replay` load |

`chat-replay` requires a dataset with structured dialogue turns. The two
`memmachine*.yaml` files use LongMemEval for this purpose. Synthetic
and Nebius configurations fail validation if paired with `chat-replay`.
Nebius currently supplies a corpus projection; coding-agent hook/MCP replay
is a separate planned scenario. See the [dataset guide](../docs/nebius.md).

`chat-profile.yaml` is an optional second YAML passed with
`--chat-profile examples/chat-profile.yaml`. It assigns users deterministically
to standard, power, and intensive groups. `concurrent_sessions` controls fixed
closed-model lanes; `max_sessions_per_user` caps active open-model sessions
and is required for every group when the profile is used with the open model.
The bundled timing variation ratios keep `answer_time` and `user_gap` inside
explicit bounded Uniform ranges instead of the default unbounded Exponential
distribution.

For `search-load`, use `--preingest` so the measured search begins with a
populated corpus. The same is recommended for `mixed` when searches should be
non-empty from the start. Each YAML header contains a complete run command and
backend-specific limitations.

See the root [README](../README.md) for scenario details and common flags.

For an offline illustration of `chat-replay` request timing, see
[Workload patterns](../docs/workload-patterns.md). The scripts and preset in
`workload-patterns/` use a fixed-delay backend and are separate from the
backend configurations above.

`memmachine-memory-types.yaml` explicitly enables type selection. All existing
REST examples remain episodic-only when these options are omitted. See
[memory types](../docs/memory-types.md) before enabling semantic processing.
