"""Command line entry point for the connector smoke checks.

Usage: ``PYTHONPATH=src:scripts python -m connector_smoke l1 --help``.
See ``scripts/connector_smoke/README.md``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TextIO

import httpx

from . import l1
from .client import SmokeClient, SmokeError

ENV_BASE_URL = "XAGENT_CONNECTOR_SMOKE_BASE_URL"
ENV_LOGIN = "XAGENT_CONNECTOR_SMOKE_LOGIN"
ENV_PASSWORD = "XAGENT_CONNECTOR_SMOKE_PASSWORD"
ENV_CALLBACK_ORIGIN = "XAGENT_CONNECTOR_SMOKE_CALLBACK_ORIGIN"
ENV_EXCLUDE = "XAGENT_CONNECTOR_SMOKE_EXCLUDE"
ENV_INCLUDE = "XAGENT_CONNECTOR_SMOKE_INCLUDE"
ENV_COOKIES = "XAGENT_CONNECTOR_SMOKE_COOKIES"
ENV_OAUTH_ACCOUNTS = "XAGENT_CONNECTOR_SMOKE_OAUTH_ACCOUNTS"


def run_l1(
    client: SmokeClient,
    *,
    login: str,
    password: str,
    login_field: str,
    login_start: bool,
    expected: set[str],
    expected_origin: str,
    oauth_accounts: Mapping[str, str],
) -> list[l1.CheckResult]:
    try:
        client.login(login, password, login_field=login_field, login_start=login_start)
    except SmokeError as exc:
        return [l1.CheckResult("login", False, str(exc))]

    try:
        apps = client.list_catalog_apps()
    except SmokeError as exc:
        return [l1.CheckResult("catalog", False, str(exc))]

    results = l1.check_catalog(apps, expected)
    for target in l1.select_oauth_targets(apps, preferred=expected):
        results.extend(
            _check_oauth_target(client, target, expected_origin, oauth_accounts)
        )
    return results


def _check_oauth_target(
    client: SmokeClient,
    target: l1.OAuthTarget,
    expected_origin: str,
    oauth_accounts: Mapping[str, str],
) -> list[l1.CheckResult]:
    provider = target.provider
    account = oauth_accounts.get(provider)
    if target.needs_account and account is None:
        return [
            l1.CheckResult(
                f"oauth:{provider}:launch",
                False,
                f"needs an account label: pass --oauth-account {provider}=<label>",
            )
        ]

    try:
        start = client.start_launch(provider, target.app_id, account)
    except SmokeError as exc:
        return [l1.CheckResult(f"oauth:{provider}:launch", False, str(exc))]
    results = [
        l1.check_callback_origin(
            provider,
            status_code=start.status_code,
            payload=start.payload,
            expected_origin=expected_origin,
        )
    ]
    authorize = f"oauth:{provider}:authorize"
    if start.status_code != 200 or not isinstance(start.payload, dict):
        results.append(l1.CheckResult(authorize, False, "launch start refused"))
        return results
    try:
        redeem = client.redeem_launch(provider, start.payload.get("authorization_url"))
    except SmokeError as exc:
        results.append(l1.CheckResult(authorize, False, str(exc)))
        return results
    results.append(
        l1.check_authorize_redirect(
            provider,
            status_code=redeem.status_code,
            location=redeem.location,
            body=redeem.body,
            expected_origin=expected_origin,
        )
    )
    return results


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    registry_rows: Iterable[Mapping[str, Any]] | None = None,
    transport: httpx.BaseTransport | None = None,
    out: TextIO | None = None,
) -> int:
    env = os.environ if env is None else env
    out = sys.stdout if out is None else out
    parser = _build_parser(env)
    args = parser.parse_args(argv)

    login = env.get(ENV_LOGIN, "")
    password = env.get(ENV_PASSWORD, "")
    if not args.base_url:
        parser.error(f"--base-url or {ENV_BASE_URL} is required")
    if not login or not password:
        parser.error(f"{ENV_LOGIN} and {ENV_PASSWORD} must be set")
    base_origin = l1.normalize_origin(args.base_url)
    if base_origin is None:
        parser.error("--base-url must be an http(s) URL")
    expected_origin = l1.normalize_origin(args.callback_origin or base_origin)
    if expected_origin is None:
        parser.error("--callback-origin must be an http(s) origin")
    try:
        cookies = _parse_pairs(
            _arg_or_env(args.cookie, env, ENV_COOKIES), separator=";"
        )
        oauth_accounts = _parse_pairs(
            _arg_or_env(args.oauth_account, env, ENV_OAUTH_ACCOUNTS), separator=","
        )
    except ValueError as exc:
        parser.error(f"{exc} (--cookie / --oauth-account)")

    if registry_rows is None:
        from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

        registry_rows = get_builtin_public_mcp_app_rows()
    try:
        expected = l1.expected_app_ids(
            registry_rows,
            exclude=_parse_list(_arg_or_env(args.exclude, env, ENV_EXCLUDE)),
            include=_parse_list(_arg_or_env(args.include, env, ENV_INCLUDE)),
        )
    except ValueError as exc:
        parser.error(str(exc))

    with SmokeClient(
        args.base_url,
        origin=base_origin,
        cookies=cookies,
        timeout=args.timeout,
        transport=transport,
    ) as client:
        results = run_l1(
            client,
            login=login,
            password=password,
            login_field=args.login_field,
            login_start=args.login_start,
            expected=expected,
            expected_origin=expected_origin,
            oauth_accounts=oauth_accounts,
        )

    failed = sum(not result.ok for result in results)
    for result in results:
        print(l1.format_result(result), file=out)
    print(f"{len(results)} checks, {failed} failed", file=out)
    return 1 if failed else 0


def _build_parser(env: Mapping[str, str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="connector_smoke")
    sub = parser.add_subparsers(dest="layer", required=True)
    p = sub.add_parser(
        "l1",
        help="configuration smoke: catalog and builtin OAuth client configuration",
    )
    p.add_argument(
        "--base-url",
        default=env.get(ENV_BASE_URL),
        help=f"deployment base URL (env {ENV_BASE_URL})",
    )
    p.add_argument(
        "--callback-origin",
        default=env.get(ENV_CALLBACK_ORIGIN),
        help="origin every OAuth callback must use; defaults to the base URL's "
        f"(env {ENV_CALLBACK_ORIGIN})",
    )
    p.add_argument(
        "--exclude",
        action="append",
        help="registry app this deployment hides; repeatable or comma-separated "
        f"(env {ENV_EXCLUDE})",
    )
    p.add_argument(
        "--include",
        action="append",
        help="app this deployment shows beyond the registry default; repeatable "
        f"or comma-separated (env {ENV_INCLUDE})",
    )
    p.add_argument(
        "--cookie",
        action="append",
        metavar="NAME=VALUE",
        help="cookie sent on every request, e.g. selected_region=sg; repeatable "
        f"(env {ENV_COOKIES}, ';'-separated)",
    )
    p.add_argument(
        "--oauth-account",
        action="append",
        metavar="PROVIDER=LABEL",
        help="account label for a provider hosted on the customer's account, "
        f"e.g. zendesk=acme; repeatable (env {ENV_OAUTH_ACCOUNTS})",
    )
    p.add_argument(
        "--login-field",
        choices=("username", "email"),
        default="username",
        help="login request field that carries the login (default: username)",
    )
    p.add_argument(
        "--login-start",
        action="store_true",
        help="call /api/auth/login/start first (Xagent SaaS region routing)",
    )
    p.add_argument("--timeout", type=float, default=20.0, help="seconds per request")
    return parser


def _arg_or_env(
    values: list[str] | None, env: Mapping[str, str], name: str
) -> list[str]:
    """Command line values replace the environment variable's, never extend it."""
    if values is not None:
        return values
    value = env.get(name, "")
    return [value] if value.strip() else []


def _parse_list(values: Iterable[str]) -> set[str]:
    return {
        item.strip() for value in values for item in value.split(",") if item.strip()
    }


def _parse_pairs(values: Iterable[str], *, separator: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for value in values:
        for item in value.split(separator):
            item = item.strip()
            if not item:
                continue
            name, sep, pair_value = item.partition("=")
            if not sep or not name.strip() or not pair_value.strip():
                # Never echo the item: a cookie value can be a credential.
                raise ValueError("expected NAME=VALUE pairs")
            pairs[name.strip()] = pair_value.strip()
    return pairs
