# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Security

- **App Widget access is now gated by an unguessable per-agent widget key (breaking)**
  The App Widget embed flow previously relied on the request `Origin`/`Referer`
  header matching an agent's `allowed_domains`. Because those headers can be set
  freely by any non-browser client (curl, scripts, a malicious server), the
  allowlist was not an effective security boundary (advisory
  GHSA-j2rx-g3hp-qg34, CWE-290). Access is now gated by a per-agent **widget
  key**:
  - The widget key is an unguessable, rotatable credential distributed inside
    the embed snippet. It is the real access gate for widget guest tokens.
  - `POST /api/widget/embed-ticket` now identifies the agent by widget key
    instead of an enumerable numeric `agent_id`. `POST /api/widget/auth` now
    requires a verifiable credential — a signed embed ticket (embedded flow) or
    the widget key (direct visit); the previous bare `Origin`/`Referer` fallback
    has been removed.
  - `allowed_domains` is retained as a **browser-level, defense-in-depth**
    restriction only. It controls which sites a real browser *without the key*
    may embed the widget from; it cannot stop a client that already holds the
    key.
  - **Residual risk:** the key is visible in the HTML of any page that embeds
    the widget, so anyone who can view such a page can extract it and mint
    low-privilege guest tokens. This is inherent to public embeddable widgets.
    Mitigations: rotate the key (revocation), the low privilege of guest tokens,
    and disabling `widget_enabled` for agents that should not be public. Note
    `widget_enabled` currently defaults to **on** for every agent, so operators
    must actively disable it where appropriate.
  - **Upgrade note:** previously deployed `data-agent-id` embed snippets stop
    working. Re-copy the embed snippet (now `data-widget-key`) from the agent's
    App Widget settings. There is intentionally no compatibility flag — the old
    path is the vulnerability.

### Added

- **Documented `LANCEDB_AUTO_MIGRATE` in environment template**
  Added `LANCEDB_AUTO_MIGRATE` usage notes to `example.env`, including default behavior (`false`) and when to enable startup auto-migration.

### Changed

- **Conversation Logs now separate historical REST API tasks from the main task list**
  Historical tasks created through the REST API (`source='sdk'`) are backfilled as hidden external conversation logs (`is_visible=false`). After migration, these tasks move out of the main task list and are available through Conversation Logs instead.

- **Knowledge Base embedding model binding (breaking / migration)**
  The Knowledge Base now treats the **Model Hub ID** as the single source of truth for embedding model identity:
  - `collection_metadata.embedding_model_id` stores the Hub ID (trimmed; no other normalization).
  - Embeddings tables are named by Hub ID: `embeddings_{to_model_tag(hub_id)}`.
  - The `model` field stored alongside each embedding vector is the Hub ID.

  **Migration / backward compatibility:** Older deployments may have created embeddings tables using the provider `model_name`
  (e.g. `embeddings_text-embedding-v4`). During search and embedding reads, the system will **try the new Hub-ID table first**
  and automatically **fall back to the legacy table name** derived from the resolved `model_name` when the new table is missing.
  Rebuild/inference helpers were updated to prefer Hub IDs when they can be resolved from Model Hub metadata.

- **Knowledge Base upload: default parse method (breaking)**
  The default parse method on the KB detail upload form is now `"default"` instead of `"pypdf"`. The backend chooses the parser by file type (e.g. .docx, .pdf). If you rely on the previous default (always use PyPDF), select `"pypdf"` explicitly in the parse method dropdown when uploading.

- **Knowledge Base document registration (breaking)**
  Document IDs for new uploads are now generated deterministically from `(collection, source_path)` instead of a random UUID. Re-uploading the same file in the same collection updates the existing record instead of creating a duplicate.
  **Impact on existing data:** Documents that were registered with the previous logic (random UUID in `doc_id`) will get a *different* `doc_id` when re-uploaded. Re-uploading such a file will create a new record rather than updating the old one, so you may briefly see two entries for the same filename until the old one is removed. If you rely on idempotent re-uploads for previously registered documents, consider deleting the old document from the KB before re-uploading, or plan a one-time cleanup of legacy duplicates.

- **LanceDB user_id migration hardening**
  Startup and migration logic now include cross-process file locking, legacy `-1` orphan marker remapping to reserved int64 sentinel values, zero-progress loop protection, and shared embeddings-table listing utilities to avoid API-compat drift.
