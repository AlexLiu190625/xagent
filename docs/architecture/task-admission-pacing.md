# Startup pacing and admission visibility

This extends durable execution admission without installing a host policy.
A host may attach `StartupPacing(interval_seconds, burst, lane)` to an
`AdmissionPolicy`. Without pacing, the existing concurrency-only behavior stays
in place. The lane is one of `batch`, `interactive`, or `default`; it is an
operational label, not a tenant identity or an authorization claim.

## Rate contract

For interval `I` and burst `B`, a newly activated bucket admits at most `B`
attempts immediately, then replenishes one start every `I` seconds. The database
stores a virtual next-start time `T`:

- Eligible when database time `now >= T - (B - 1) * I`.
- On a committed new reservation, set `T = max(now, T) + I`.

This is a continuous allowance, with no fixed-window reset. Idle time restores
at most `B` immediate starts. A backwards database clock jump delays admission;
it does not mint starts. All workers use the same database clock and serialize
through the same bucket row as concurrency admission. Worker-local wall clocks
cannot inflate the allowance.

The token is consumed in the command-claim transaction and rolls back if the
claim does not commit. It represents an admitted execution attempt, not a model
request. Failed/recovered attempts that reacquire a slot also acquire a start.
A claim resumed under the same owner and same-bucket live MESSAGE guidance do
not consume an additional start. Cancelling or completing an execution does
not refund an already consumed start.

A guidance claim cannot silently become a new execution if its original run
finishes before routing. The actual RUNNING transition checks its admission;
unadmitted transitions roll back and return the command to durable waiting.
The first-party resume handoff makes this decision before spawning background
work. Only a confirmed injection may continue its exact original run without a
new start. A waiting command retains its delivery identity and monotonic claim
counter (a write fence), while failure and deferral counters remain unchanged.
Failed guidance joins retry after 1, 2, 4, 8, 16, 32, then at most 60 seconds.
The backoff does not prevent queued cancellation or pause from overtaking them.

Hosts reserve both concurrency and startup allowance by assigning separate
batch and interactive buckets. Neither bucket consumes the other's allowance.
A batch backlog cannot delay eligible interactive work through this rate gate.
The dispatcher, database, and target task must still be healthy; provider
first-token latency is outside this contract. This is not the provider's
30-minute load-growth policy and does not bound DAG-internal model concurrency.

Pacing values are explicit and positive; no numeric provider-safe default is
inferred. To change an existing paced bucket, stop ingress, drain existing work,
and update persisted policy and all host configuration together. Do not roll
activation across old executors. Upgrade to head
(`20260925_merge_admission_pacing`) first; disable producers and drain before
downgrade.

## Visibility and bounded attribution

`read_admission_snapshot(session, authorized_bucket_keys)` accepts at most 32
explicit keys. Hosts own authorization and team-to-key resolution; there is no
public enumerate-all-tenants API. Each result exposes:

- configured capacity and pending budget, active reservations and queue depth;
- age since acceptance of the oldest unresolved waiting command;
- startup interval, burst and remaining startup delay;
- the current bucket delay reason: capacity, startup pacing, or dispatch.

Snapshots read persisted ownership, so an expired owner still consumes capacity
until the existing recovery protocol retires its token. Settled business state
alone does not make a running finalizer invisible. Results are operational
observations; another worker may change them immediately after the read.

OpenTelemetry uses the existing runtime exporter:

| Instrument | Meaning |
| --- | --- |
| `task.admission.claims` | Committed ticketed claims, outcome `initial` or `retry_or_recovery`. |
| `task.admission.initial_wait` | Seconds from durable acceptance to the first committed claim. |
| `task.admission.queue_full` | Ingress or explicit-retry refusals due to the pending budget. |

Only bounded lane/outcome labels are recorded (`operation` carries the lane).
Team/task/command IDs, payloads, credentials and bucket names are never metric
labels. Team attribution comes from authenticated scoped snapshots. Metrics are
best-effort operational counters, not a billing ledger; an exporter failure
cannot reject admitted work. Initial-wait excludes retry/recovery attempts rather
than mixing previous execution time into queue-wait latency.

The shared client error table recognizes `execution_queue_full` and provides
English/Chinese retry wording for hosts that emit that code. Host acceptance and
quota semantics belong to the SaaS integration, not this engine extension.

## Verification

Tests drive durable ingress and dispatch on SQLite and PostgreSQL. Deterministic
clock inputs verify burst edges, exact refill boundaries, long idle intervals,
backwards clocks and independent interactive allowance. Tests also cover live
guidance, persisted observation and bounded metric labels. Three independent
processes race for one startup budget using the real database clock. Migration
checks preserve existing tasks and capacity configuration on upgrade/downgrade.
