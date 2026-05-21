# E2E Tests

## Running The Tests

Run all e2e tests:

```bash
uv run --project . --group test python -m pytest tests/e2e --run-special -q
```

## File Persistence E2E Expected Behavior

This document maps the expected durable file storage behavior to the e2e tests that cover it.

The durable file storage implementation is a first-phase, local-first persistence
layer with casual consistency across local disk, the `uploaded_files` table, and
S3/MinIO. It does not try to provide distributed transaction semantics. The
normal contract is:

- Local files are preferred when present.
- Durable storage is the fallback when the registered local file is missing.
- DB rows are marked `storage_status = "available"` only after durable storage
  returns object metadata with a checksum.
- Startup repair reconciles DB-registered files into durable storage where local
  bytes or existing durable objects make repair possible.
- Transient durable-storage failures should fail closed and be retried; known
  unrecoverable local+remote data loss is surfaced later as a missing file rather
  than blocking all app startup.

### Expected Behavior And Coverage

#### Startup Sync

- App startup scans DB-registered files and uploads legacy local files to S3/MinIO when durable metadata is missing.
  Covered by: `tests/e2e/test_file_startup_sync_minio.py::test_startup_sync_repairs_only_files_that_need_durable_storage`
- App startup does not overwrite an object that already exists in S3/MinIO when DB durable metadata is complete.
  Covered by: `tests/e2e/test_file_startup_sync_minio.py::test_startup_sync_repairs_only_files_that_need_durable_storage`
- App startup repairs rows whose DB metadata says S3 is available but the remote object is missing, as long as the local file still exists.
  Covered by: `tests/e2e/test_file_startup_sync_minio.py::test_startup_sync_repairs_only_files_that_need_durable_storage`
- App startup skips rows whose local file is missing and remote object is missing, without failing app startup.
  Covered by: `tests/e2e/test_file_startup_sync_minio.py::test_startup_sync_repairs_only_files_that_need_durable_storage`

#### Upload And API Persistence

- User file upload writes the local file, creates an `uploaded_files` DB row, and persists the object to S3/MinIO.
  Covered by: `tests/e2e/test_file_api_minio.py::test_download_and_preview_materialize_uploaded_file_from_minio`, `tests/e2e/test_file_persistence_minio.py::test_task_uploads_agent_outputs_and_startup_sync_persist_to_minio`

#### Download And Preview

- Download can recover a missing local file from S3/MinIO and restore it to the registered local path.
  Covered by: `tests/e2e/test_file_api_minio.py::test_download_and_preview_materialize_uploaded_file_from_minio`
- Preview can serve a file from S3/MinIO when the registered local file is missing.
  Covered by: `tests/e2e/test_file_api_minio.py::test_download_and_preview_materialize_uploaded_file_from_minio`

#### Delete And Access Control

- Deleting a file removes the DB row, local file, and S3/MinIO object.
  Covered by: `tests/e2e/test_file_api_minio.py::test_delete_removes_uploaded_file_from_db_local_disk_and_minio`
- A durable storage cleanup failure fails the delete request with HTTP 503 and
  keeps the DB row, local file, and S3/MinIO object so the delete can be retried.
  Covered by: `tests/e2e/test_file_api_minio.py::test_delete_keeps_db_row_when_durable_cleanup_fails`
- Another user cannot download, preview, or delete a file they do not own, and the S3/MinIO object remains intact.
  Covered by: `tests/e2e/test_file_api_minio.py::test_file_routes_reject_cross_user_access_and_keep_minio_object`

#### WebSocket And Agent File Persistence

- WebSocket task execution persists agent-created output files to DB and S3/MinIO with output workspace metadata.
  Covered by: `tests/e2e/test_file_persistence_minio.py::test_task_uploads_agent_outputs_and_startup_sync_persist_to_minio`
- Chat/WebSocket task execution can materialize a missing local input from S3/MinIO before the agent reads it, then persist derived output to S3/MinIO.
  Covered by: `tests/e2e/test_file_persistence_minio.py::test_chat_task_materializes_missing_local_input_from_minio_before_agent_reads_it`

#### Remote Storage Outage Behavior

- User file upload fails with HTTP 503, removes temporary local files, and leaves no `uploaded_files` row when durable storage write fails.
  Covered by: `tests/e2e/test_file_api_minio.py::test_upload_returns_503_and_rolls_back_when_minio_write_fails`
- Download serves an existing local copy during durable storage outage, but returns HTTP 503 when the local copy is missing and durable storage cannot be read.
  Covered by: `tests/e2e/test_file_api_minio.py::test_download_serves_local_copy_when_minio_read_fails`, `tests/e2e/test_file_api_minio.py::test_download_and_preview_return_503_when_minio_read_fails_without_local_copy`
- Preview returns HTTP 503 when the local copy is missing and durable storage cannot be materialized.
  Covered by: `tests/e2e/test_file_api_minio.py::test_download_and_preview_return_503_when_minio_read_fails_without_local_copy`
- WebSocket output persistence rolls back the output DB row and fails task output normalization when durable storage write fails.
  Covered by: `tests/e2e/test_file_persistence_minio.py::test_websocket_output_persistence_sends_error_and_rolls_back_when_minio_write_fails`

### Supporting Harness Coverage

- E2E JWT helper creates a token with expected user claims.
  Covered by: `tests/e2e/test_app_harness.py::test_build_access_token_contains_user_claims`
- E2E local file seeding creates a physical file and matching `uploaded_files` row.
  Covered by: `tests/e2e/test_app_harness.py::test_seed_registered_local_file_creates_file_and_db_record`
- Scripted LLM JSON fixtures are converted into mock LLM responses, including native tool-call payloads.
  Covered by: `tests/e2e/test_scripted_llm.py::test_load_scripted_responses_converts_enveloped_entries`

### Current Boundaries

- These tests use the real FastAPI app startup path through `TestClient`.
- S3 behavior is exercised against Docker MinIO.
- LLM behavior is deterministic through `tests/e2e/scripted_llm.py` and JSON response fixtures.
- These tests do not cover a full remote S3 outage during app startup; that should be covered by a smaller service/integration test if needed.
- E2E outage tests cover upload, download, preview, delete, and WebSocket output persistence under durable storage failures by monkeypatching the storage layer while using the real FastAPI app, DB, and Docker MinIO configuration.
- Delete outage tests assert fail-closed behavior: user-facing delete does not
  remove the DB row when durable cleanup fails, and stale KB reconciliation can
  retry cleanup later.

### Production Durability Gaps (TODO)

These e2e tests verify the normal durable-storage contract, but they do not prove all production durability failure modes.

- Crash consistency is not covered. For example, a process crash after writing an object to S3/MinIO but before committing the `uploaded_files` row could leave an orphan durable object; a crash after DB metadata is committed but before local cleanup or response completion could leave partial local state.
- Read integrity verification is not covered. The storage layer records checksum metadata on write, but these tests do not assert checksum validation when materializing or downloading from durable storage.
- Concurrent writer and multi-process repair behavior is not covered beyond startup lock acquisition. Tests should cover duplicate uploads, repeated startup sync, and races between startup repair, user download, and delete.
- Object lifecycle cleanup is not covered. A process crash or late failure can
  still leave orphaned S3/MinIO objects, and these tests do not assert cleanup or
  reconciliation for objects that have no committed DB row.
- S3 checksum metadata repair is not covered. If object bytes are written but
  checksum metadata attachment fails, startup repair may see the object but
  refuse to mark the DB row available until metadata can be inspected or the
  object is manually repaired.
