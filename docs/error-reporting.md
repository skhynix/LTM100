# Operation error diagnostics

Failed ADD/SEARCH operations carry bounded diagnostics in `raw.ndjson`
(`--raw`) and an `error_breakdown` in `summary.json`. No new CLI option or
backend configuration is required. The JSON printed without `--output` also
contains the breakdown. Existing fields and `summary.csv` keep their meaning
and format; successful and rejected raw records have no `error_detail` field.

## Per-operation records

An HTTP failure can produce:

```json
{
  "op_type": "search",
  "user_id": "syn_user_00000",
  "status": "error",
  "error_kind": "RestError",
  "error_detail": {
    "category": "http",
    "message": "HTTP 429 Too Many Requests",
    "method": "POST",
    "endpoint": "/api/v2/memories/search",
    "http_status": 429
  }
}
```

This excerpt omits the existing timestamps, latency, and item count. Those
fields are still recorded. The original exception class stays in
`error_kind`; the new category supplements it.

| Category | Meaning |
|---|---|
| `http` | The REST server returned an HTTP error status |
| `timeout` | A recognized timeout exception occurred |
| `connection` | A recognized connection failure occurred |
| `response` | JSON decoding or REST response-body transfer failed |
| `mcp` | An MCP call failed without a more specific recognized cause |
| `other` | An operation failed without a recognized transport category |

Only available fields are included. REST failures include the method and
endpoint path. A JSON decoding failure can include `http_status: 200`: the
HTTP request succeeded, but its body was invalid JSON. `timeout_seconds` is
the configured request limit when available, **not** measured latency or a
server-side execution deadline. No HTTP status is inferred from an exception
message or from a timeout.

MCP failures include the tool name and endpoint path. Recognized explicit
causes, such as `TimeoutError`, remain visible through the wrapper's category.
A failing tool response's native `status` becomes `tool_status`, never an
invented HTTP status. Unknown SDK failures retain their existing exception
class and use the best available category; messages are not parsed to guess
their cause.

## Aggregation

`summary.error_breakdown` groups failed operations by operation type,
exception class, category, and available method/path/status/tool fields:

```json
{
  "errors_by_kind": {"RestError": 12},
  "error_breakdown": [
    {
      "op_type": "search",
      "error_kind": "RestError",
      "category": "http",
      "method": "POST",
      "endpoint": "/api/v2/memories/search",
      "http_status": 429,
      "count": 12
    }
  ]
}
```

The same breakdown is included per operation and, for profiles, per group.
Counts are pooled from recorded operations across processes. Messages and
timeout settings are not grouping keys. A summary without detailed failures
has no `error_breakdown`, including legacy core callers that construct
`OpResult` without diagnostics. If detailed and legacy failures are mixed,
the legacy failures join the breakdown as `other` so its count still matches
`errors`.

## Measurement and compatibility

- One failed logical operation is still one error, even if the adapter sends
  multiple HTTP requests or the transport retries. Details describe its final
  failure; earlier attempts and partially completed writes are not audited.
- HTTP status failures are not retried by the REST transport. Connection and
  timeout retries retain the existing configured count and backoff.
- Request completion is timestamped before runner-side diagnostic extraction.
  Diagnostic extraction is bounded synchronous work on failed calls; it does
  not make network requests, sleep, or change workload policy.
- Successful throughput and latency, error-rate denominators, and empty-result
  metrics retain the [existing report contract](running.md#reports).
- Queue rejections remain `status: rejected`, `error_kind: queue_full`; they
  are not backend errors and do not enter `error_breakdown`. Cancellation
  still propagates and is not converted into a failed operation.
- Setup, pre-ingest, and cleanup retain their existing exception behavior.
  Their failures are not added to measured operation counts. Warm-up results
  remain excluded. Optional server-metrics scrape warnings remain separate
  from workload operation diagnostics.

## Data boundaries

Operation diagnostic fields use generated messages, not `str(exception)`.
They do not include response bodies, request bodies, headers, authentication, tool
arguments, hostnames, URL credentials, query parameters, or fragments.
Endpoint paths are limited to 256 characters and tool names to 80. Paths may
still contain resource identifiers when those are part of the API route.
Existing exception strings, CLI traceback behavior, and debug logging are
unchanged; this is not a general log-redaction feature.

Diagnostics preserve exception identity/type and explicit causes. If an
exception cannot carry metadata, recording falls back to a safe category.
