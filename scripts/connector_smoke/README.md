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
