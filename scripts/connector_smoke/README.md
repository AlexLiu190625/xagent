# Connector smoke checks

Black-box HTTP checks for connectors on a deployed Xagent instance, tracked in
[#3244](https://github.com/xorbitsai/xagent/issues/3244). They log in as a
dedicated low-privilege monitor account and call only public routes. They never
run an LLM agent.

## L1: configuration smoke

L1 needs no vendor accounts. It checks two things.

**Catalog.** `GET /api/mcp/apps` must list every expected app, and each one must
be connectable (its `auth_type` is not `unconnectable`). Any app the catalog
lists that is not expected also fails. The expected set is:

- every app that the builtin registry (`src/xagent/web/builtin_mcp_registry.py`)
  shows by default,
- minus the apps this deployment hides (`--exclude`),
- plus the apps this deployment shows beyond that default (`--include`).

New connectors are picked up from the registry without editing any list. An app
an administrator created is not in the registry, so it fails as unexpected
until you pass it with `--include`.

The registry decides Gmail's and full Google Drive's default visibility from
`XAGENT_GOOGLE_RESTRICTED_SCOPES_ENABLED` in the environment of the process
running the check, not the deployment's. Leave it unset where the check runs.
Against a deployment that has the gate open, pass `--include gmail,google-drive`.

**Builtin OAuth providers.** The check launches each provider once, through one
of its visible catalog apps (a registry app when there is one):

1. `POST /api/auth/{provider}/login` starts the launch. The returned callback
   origin must equal the environment's origin. An unset `<PROVIDER>_REDIRECT_URI`
   falls back to `http://localhost:8000`, so it fails here.
2. `GET` on the returned launch path redeems it, without following the
   redirect. The provider `Location` must carry a non-empty `client_id` and a
   `redirect_uri` of `<origin>/api/auth/{provider}/callback`.

Remote MCP OAuth apps (Notion, Atlassian, HubSpot, …) are not launched, because
their connect may register a client with the vendor on every call. L2 covers
them.

L1 cannot detect a missing client secret: the server first reads the secret in
the OAuth callback, after the vendor's consent screen. Missing secrets are
covered by `python -m xagent.web.oauth_readiness` instead
([#3248](https://github.com/xorbitsai/xagent/issues/3248)). Run it in the
backend container through the image entrypoint, which loads `ENCRYPTION_KEY`
from its key file the same way it does for the server:

```bash
docker compose exec -T backend /opt/xagent/deploy/entrypoint.sh \
  python -m xagent.web.oauth_readiness
```

It prints one line per provider saying whether the client ID and secret
resolve, and from the `oauth_providers` row or the environment, never a value.
A row encrypted under another key fails as `unreadable`; a process with no
usable key fails once as `encryption-key`. It proves a value is present, not
that the vendor accepts it. Its `--exclude` takes OAuth provider names
(`google`), not the catalog app IDs this script's `--exclude` takes.

### Side effects

- Starting a launch writes nothing.
- Redeeming one stores a single-use ticket record (`oidc_consumed_tokens`,
  `oauth-launch:<id>`) that expires after 30 seconds. Every redeem deletes the
  records that have already expired, so they do not pile up across runs: a run
  more than 30 seconds after the previous one leaves one row per provider.
- Logging in replaces the monitor account's refresh token. Give the check its
  own account; do not share one that another client keeps a session for.

### Usage

Run from the repository root:

```bash
export XAGENT_CONNECTOR_SMOKE_LOGIN=monitor@example.com
export XAGENT_CONNECTOR_SMOKE_PASSWORD=...
PYTHONPATH=src:scripts python -m connector_smoke l1 --base-url https://xagent.example.com
```

On Xagent SaaS, pick the region with a cookie, start the login through
`login/start`, and log in by email. `selected_region` is sent on every request,
and `login_route` is sent only on the login request:

```bash
PYTHONPATH=src:scripts python -m connector_smoke l1 \
  --base-url https://cloud.xagent.co \
  --cookie selected_region=sg \
  --login-start --login-field email \
  --oauth-account zendesk=<subdomain> --include zendesk
```

| Option | Environment variable | Meaning |
| --- | --- | --- |
| `--base-url` | `XAGENT_CONNECTOR_SMOKE_BASE_URL` | Deployment base URL. Required. |
| (none) | `XAGENT_CONNECTOR_SMOKE_LOGIN`, `XAGENT_CONNECTOR_SMOKE_PASSWORD` | Monitor account. Accepted only from the environment. |
| `--callback-origin` | `XAGENT_CONNECTOR_SMOKE_CALLBACK_ORIGIN` | Origin every OAuth callback must use. Defaults to the base URL's origin. |
| `--exclude` | `XAGENT_CONNECTOR_SMOKE_EXCLUDE` | Registry apps this deployment hides. Each one must be shown by the registry by default. |
| `--include` | `XAGENT_CONNECTOR_SMOKE_INCLUDE` | Apps this deployment shows beyond the registry default. |
| `--cookie NAME=VALUE` | `XAGENT_CONNECTOR_SMOKE_COOKIES` (`;`-separated) | Cookies sent on every request. |
| `--oauth-account PROVIDER=LABEL` | `XAGENT_CONNECTOR_SMOKE_OAUTH_ACCOUNTS` | Account label for a provider hosted on the customer's own account, for example Zendesk. The label must resolve, so use a real account. |
| `--login-field` | (none) | `username` (default) or `email`. |
| `--login-start` | (none) | Call `/api/auth/login/start` before logging in. |

The list options are repeatable or comma-separated. A value given on the command
line replaces the environment variable's value; it does not add to it.

### Output

The check prints one line per check, then a summary:

```text
PASS catalog:slack
PASS oauth:slack:callback
FAIL oauth:slack:authorize: no provider redirect: HTTP 500 (Error: OAuth provider not configured - Missing SLACK_CLIENT_ID for provider slack.)
42 checks, 1 failed
```

The exit code is `0` when every check passes, `1` when any check fails, and `2`
for a usage error. The output never contains tokens, tickets, client IDs, or
redirect URLs.

### Tests

```bash
pytest tests/scripts/test_connector_smoke_l1.py tests/scripts/test_connector_smoke_cli.py
```

The tests use recorded response shapes and an in-process fake deployment. None
of them opens a network connection.

## Startability check: every connector, no accounts

L1 proves a connector is listed and its OAuth client is configured; it cannot
tell whether the connector starts. `python -m xagent.web.connector_readiness`
([#3264](https://github.com/xorbitsai/xagent/issues/3264)) does, for every
`public_mcp_apps` row of the deployment, hidden rows included, without any
vendor account. Run it in the backend container the same way as
`oauth_readiness`:

```bash
docker compose exec -T backend /opt/xagent/deploy/entrypoint.sh \
  python -m xagent.web.connector_readiness
```

What it checks depends on the row's `launch_config`:

| Shape | Check |
| --- | --- |
| `command` (`python -m xagent.web.tools.mcp.<app>`) | Spawns the command with a placeholder for every `env_mapping` and `required_env` key, runs MCP `initialize` and `tools/list`, and requires at least one tool. The in-repo servers' tool lists are static, and the check stops after `tools/list`, so the placeholder never reaches a vendor. |
| `command` `npx`/`uvx` | `SKIP sandboxed` when `SANDBOX_ENABLED` is on, because tasks run these in the sandbox image, not the backend image. With the sandbox off, spawned as above with npm and uv forced offline, so a package missing from the image's cache fails instead of being fetched from a registry. |
| `url` (remote MCP, e.g. Notion, Atlassian, HubSpot) | One unauthenticated `GET` through the runtime's MCP HTTP client, so the probe sees the same proxy and egress policy a task does. Any HTTP response passes. No OAuth flow is started. |
| `url` with a `ws://`/`wss://` scheme | `SKIP websocket`; websocket rows are not probed. |

It prints one line per connector, labelled `builtin` (a registry app) or
`custom` (an administrator-created row), plus `hidden` for a row the catalog
does not show, then a summary:

```text
PASS connector:slack builtin: 11 tools
SKIP sandboxed connector:chrome-devtools builtin hidden: tasks run npx/uvx rows in the sandbox image, not this one
SKIP sandboxed connector:google-maps builtin: tasks run npx/uvx rows in the sandbox image, not this one
FAIL missing connector:xero builtin: /usr/local/bin/python: No module named xagent.web.tools.mcp.xero
FAIL unreachable connector:notion builtin: ConnectError: [Errno -3] Temporary failure in name resolution
51 checks, 2 failed, 2 skipped
```

A failure's class is one of `spawn` (command not found, or a row with neither
a command nor a url), `missing` (the child exited before the MCP handshake: a
Python import error, or an npm package absent from the offline cache),
`handshake` (no usable `initialize`/`tools/list` reply within the timeout, an
error reply, or a connector not reached before the total bound), `no-tools`, or `unreachable`. The
detail is the last line of the child's stderr or the exception, never the
environment. When the database cannot be read, `FAIL database: <ExceptionClass>`
is printed once and the run exits `1`; when no row is checked (an empty
catalog, or every row excluded), `FAIL catalog: no public_mcp_apps rows to
check` is printed and the run exits `1`. The exit code is also `1` when any
connector fails and `2` for a usage error.

Each connector is bounded by `XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS` (60 s by
default, the same bound task startup uses) and the whole run by
`--total-timeout`, which defaults to the connector count times that value; a
connector not reached in time fails as `handshake`.
`XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS=0` disables both bounds, and
`--total-timeout 0` disables the total bound. `--exclude` takes catalog app
IDs, like this script's `--exclude`, for a connector the deployment leaves
unstartable on purpose. On a developer machine the 44 python-module rows took
12 seconds in total; an npx row that resolves offline and a remote probe add a
few seconds each.

What a pass proves: the module or package is present in the backend image and
answers the handshake, or the remote host answers HTTP from the backend
network. It does not prove a credential works (L3) or that an `npx` row
starts in the sandbox image (#1692).

```bash
pytest tests/web/test_connector_readiness.py
```

The tests spawn fake stdio servers with the test interpreter and replace the
HTTP client; none of them opens a network connection.
