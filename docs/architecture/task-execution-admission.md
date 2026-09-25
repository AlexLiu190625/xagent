# Durable execution admission

Shared hosts can limit active execution by registering
`set_task_admission_hook`. The hook runs inside new command acceptance and
returns `AdmissionPolicy(bucket, capacity, max_pending)`, or `None` for an
ungoverned command. Without a hook, no admission tickets are created and
existing execution policy remains unchanged. This foundation does not install
a tenant policy, change billing, add a public request field, or pace starts.

The host must register its classifier consistently on every ingress/worker
that can create commands. Classification is server-owned and stored once,
with the command, in the same transaction. Retrying a command ID does not
reclassify it or consume another queue position. A classifier exception or
`AdmissionQueueFull` must abort the caller's acceptance transaction. The host
translates queue-full into its retryable public error. Local execution is not
an admission topology: selecting a policy with shared execution disabled fails.
Classifiers must be fast and free of external side effects: they run inside
acceptance before the bucket-row write serializes acceptance, claim, and retry.

## Budget and ordering

Each opaque bucket has a positive capacity and pending-command budget. Hosts
can partition a team's capacity into separate batch and interactive buckets;
those buckets do not borrow from one another. Tenant identity, traffic class,
numeric defaults, quota semantics, and ingress coverage belong to the host
integration. A task-origin label alone is not a trusted classification of a
later turn.

Workers first exclude capacity-blocked candidates from their scan, then lock
the bucket and recheck before committing a command claim. PostgreSQL row
locking and SQLite write transactions serialize admissions across processes.
The pending budget is checked under that same bucket lock at acceptance.
It bounds waiting commands, not active executions; crash recovery can return
already-accepted active work to waiting without discarding it.

Waiting commands keep their existing pending/processing state and do not spend
attempt, failure, or defer budgets merely because capacity is unavailable.
Oldest waiting command ID wins within a bucket, including for prompt dispatch
by ID. A busy earlier task or a business-retry deadline can therefore delay
later work in that same bucket. Separate buckets remain independently eligible.
The existing per-task ordering and command authorization still apply.
Admission waiting has no automatic expiry. Waiting alone does not exhaust a
retry budget; work leaves the queue through execution, terminal control/failure,
or command/task deletion.

Policy values are immutable for an existing bucket in this first foundation.
Mismatched values fail acceptance rather than letting workers use conflicting
limits. To change them operationally, stop producers, drain execution and
pending work, and update the persisted policy and host configuration together
before restarting. A future administration interface can own that procedure.

## Ownership and release

A claimed ticket belongs to the exact `(task_id, runner_id, owner_attempt_id)`
of its coordinator. It consumes capacity until the actual execution handles
and their cleanup have drained. Publishing COMPLETED, PAUSED, or
WAITING_FOR_USER alone does not release the slot. Before the same owner begins
another command, a drained, non-running previous execution releases its tickets.
Tool waits and model retries keep their active slot. MESSAGE guidance delivered
to a running execution in the same bucket joins its existing slot. It still
obeys the pending-command budget at acceptance. A differently classified
MESSAGE reserves its own bucket conservatively until execution cleanup; it
cannot borrow another bucket's slot if routing changes to a new turn.

Lease expiry alone is not capacity recovery. The existing recovery path must
retire the owner token first. A live successor does not inherit the old
reservation, and a late release carrying the old token cannot free its slot.
Crash recovery preserves the existing task execution semantics: this change
does not replay completed START handoffs or promise exactly-once external tool
side effects.

Completed tickets are removed on normal release. Failed commands retain their
classification so an explicit retry rechecks the pending budget and waits for
capacity again. Existing command rows retain idempotency and outcome records. Abrupt
owner loss leaves inactive ticket evidence until recovery or ordinary command/
task deletion; foreign-key cascades clean up the dependent ticket rows.

## Controls

CANCEL and PAUSE do not acquire execution capacity. They may overtake only
older pending commands that have not reserved a slot. Their existing executor
still validates ownership, authorization, run and state version. Successful
controls that change the target state invalidate earlier waiting commands for
that old state and persist their terminal events before committing completion. A rejected or ineffective
control does not discard queued work. Active execution cancellation retains its
slot until its cleanup actually finishes.

A PAUSE aimed at the future run of an unreserved START settles that exact START
and the PAUSE atomically, before the START can schedule after capacity opens.
It requires the live claim, current identities, unchanged run/version, and an
unreserved ticket. The terminal START retains classification for explicit retry.
A never-started task uses the existing non-resumable FAILED status with a
user-stop reason; it has no execution checkpoint to resume as PAUSED. Stopping
an appended turn preserves the previous run's status and answer.

## Matching and trust boundaries

| Axis | Gate |
| --- | --- |
| Stable ID | Command ID and task ID bind the ticket; runner ID plus owner attempt fence release. |
| Name | Display names do not participate in identity or capacity accounting. |
| Transport | Shared START, RESUME, RESUME_INPUT and MESSAGE commands use the same staging and claim gates; controls are excluded. |
| Launch/configuration | Shared execution is required; hosts must install one consistent classifier and policy. |
| Authentication | Admission adds no authority; the existing command executor retains its actor and target checks. |
| Ownership | Expiry does not transfer a reservation; recovery must retire the old acquisition. |
| Scope | The host constructs stable bucket keys; client payloads must not choose or elevate their own lane. |
| Credentials | No credentials or runtime secret values are stored in tickets. |

## Verification and upgrade

The integration suite exercises durable ingress and dispatch on SQLite and
PostgreSQL, including three separate worker processes, pending-budget refusal,
FIFO prompt dispatch, cancellation, continuation commands, cleanup, and owner
recovery. Executors in these tests control completion timing; no live model is
required. Existing start/resume/coordinator tests cover their execution wiring.

Upgrade to head (`20260924_merge_admission_google`) before enabling the host policy
and run the same version on every executor. Resolve the known FIFO eligibility
limits ([#2648](https://github.com/xorbitsai/xagent/issues/2648)) and recovered/
attempted-command control classification
([#2649](https://github.com/xorbitsai/xagent/issues/2649)) before activation.
Old binaries do not enforce admission tickets;
rolling activation with old executors is unsupported. Disable producers and
drain admitted work before downgrade. The migration adds only admission tables
and does not rewrite task or command state.
