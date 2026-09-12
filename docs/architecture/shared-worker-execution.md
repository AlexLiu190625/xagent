# Shared task execution foundations

This change contains the acceptance/data and event/recovery foundations for shared task execution. Application startup and task execution remain on the existing path. It does not start a shared worker or the Redis event bridge.

## Acceptance and persistence

`_AcceptedTurn` represents a persisted turn without an execution lease. The existing `_claim_turn_no_commit` still acquires the lease and returns `_ClaimedTurn` inside the same transaction before scheduling. Message and file binding, Workforce projection, commit reconciliation and local scheduling keep their existing contracts.

`task_runtime_secrets` stores encrypted, single-turn connector secrets and auth selectors. Reads verify task, turn, run and the owner's stable subject. Storage, binding and cleanup operations are available to later ingress/worker integration; existing connector runtime paths do not use this store yet. No secret values are added to task APIs or event payloads.

Nullable `reply_host_id` and `reply_origin` fields identify the creating ingress and its exact socket registration. They are stored when a command is created; duplicate submissions cannot overwrite the route. The migration preserves existing commands and can be downgraded independently.

## Events and recovery

The Redis bridge provides task fanout, origin-bound private replies, socket acknowledgements, bounded deduplication and reconnect notifications. Redis remains a Pub/Sub transport, not a task queue or replay log. A private-reply timeout is delivery-unknown and must not cause command execution to retry.

State-bearing event enrichment is extracted from the WebSocket API so a future producer can capture the original run/version before transport. Each shared socket has a bounded writer queue; slow sockets cannot block the other recipients. Authorized task audiences can receive persistent state/output snapshots to reconcile missing events.

The frontend handles unavailable/resync notifications and exact-run snapshots. It marks interrupted output, avoids appending tokens to a missing prefix, and replaces the display when complete persisted output is available. Existing local output remains unchanged when these events are absent.

## Activation boundary

`XAGENT_SHARED_TASK_EXECUTION_ENABLED` defaults to **false in this intermediate change**. Keep it false: this change does not wire the bridge into application startup and is not a deployable shared execution mode. Tests explicitly initialize the bridge to exercise its contracts.

`XAGENT_TASK_EVENT_CHANNEL_PREFIX` defaults to `xagent:task-events:v1`. Redis logical DB numbers do not isolate Pub/Sub; separate deployments need different prefixes.

Follow-up changes will implement START handoff and lease fencing, resume/existing-task and ingress adapters, channel bots, and worker/web lifecycle integration. The final integration will enable shared execution by default, after those paths are complete. No intermediate PR should turn it on prematurely.

## Validation

Tests accompany acceptance/rollback, immutable origin routing, encrypted input isolation, migration upgrade/downgrade, event identity and deduplication, ACK semantics, bounded socket queues, and frontend recovery. The Redis reconnect test accepts `XAGENT_TEST_REDIS_URL`; PostgreSQL migration tests accept `XAGENT_TEST_POSTGRES_URL` and create/drop their own disposable database. Missing service URLs skip only those external-service tests.
