#!/usr/bin/env python3
"""Refresh open migration PRs after Alembic changes land on main."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

BASE_BRANCH = os.getenv("BASE_BRANCH", "main")
MIGRATION_PREFIX = os.getenv("MIGRATION_PREFIX", "src/xagent/migrations/")
MIGRATION_VERSIONS_DIR = os.getenv(
    "MIGRATION_VERSIONS_DIR", "src/xagent/migrations/versions"
)
DISPATCH_WORKFLOWS = tuple(
    workflow.strip()
    for workflow in os.getenv("DISPATCH_WORKFLOWS", "").split(",")
    if workflow.strip()
)
STATUS_CONTEXT = os.getenv("STATUS_CONTEXT", "migration-refresh/alembic-heads")
GITHUB_API_URL = os.getenv("GITHUB_API_URL", "https://api.github.com")
GITHUB_SERVER_URL = os.getenv("GITHUB_SERVER_URL", "https://github.com")


class GitHubError(RuntimeError):
    """Raised when a GitHub API call fails."""

    def __init__(self, method: str, path: str, status: int, body: str):
        super().__init__(f"{method} {path} failed with {status}: {body}")
        self.method = method
        self.path = path
        self.status = status
        self.body = body


class GitHub:
    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        attempts = 3 if method == "GET" else 1
        last_error: GitHubError | None = None

        for attempt in range(attempts):
            body = None
            if payload is not None:
                body = json.dumps(payload).encode("utf-8")

            req = urllib.request.Request(
                f"{GITHUB_API_URL}{path}",
                data=body,
                method=method,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {self.token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            if body is not None:
                req.add_header("Content-Type", "application/json")

            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    response_body = response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = GitHubError(method, path, exc.code, error_body)
                if (
                    method == "GET"
                    and exc.code in {429, 500, 502, 503, 504}
                    and attempt < attempts - 1
                ):
                    time.sleep(2**attempt)
                    continue
                raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = GitHubError(method, path, 0, str(exc))
                if method == "GET" and attempt < attempts - 1:
                    time.sleep(2**attempt)
                    continue
                raise last_error from exc

            if not response_body:
                return None
            return json.loads(response_body)

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"{method} {path} failed without a response")

    def paginate(self, path: str) -> list[Any]:
        separator = "&" if "?" in path else "?"
        page = 1
        results: list[Any] = []

        while True:
            page_path = f"{path}{separator}per_page=100&page={page}"
            batch = self.request("GET", page_path)
            if not isinstance(batch, list):
                raise RuntimeError(f"Expected paginated list from {path}")
            results.extend(batch)
            if len(batch) < 100:
                return results
            page += 1

    def status_url(self) -> str | None:
        run_id = os.getenv("GITHUB_RUN_ID")
        if not run_id:
            return None
        return f"{GITHUB_SERVER_URL}/{self.repo}/actions/runs/{run_id}"

    def set_status(self, sha: str, state: str, description: str) -> None:
        payload: dict[str, Any] = {
            "state": state,
            "context": STATUS_CONTEXT,
            "description": description[:140],
        }
        target_url = self.status_url()
        if target_url:
            payload["target_url"] = target_url
        self.request("POST", f"/repos/{self.repo}/statuses/{sha}", payload)


@dataclass(frozen=True)
class Revision:
    revision: str
    down_revisions: tuple[str, ...]
    path: str


def _literal_revisions(value: ast.AST | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, ast.Constant):
        if value.value is None:
            return ()
        if isinstance(value.value, str):
            return (value.value,)
    if isinstance(value, (ast.Tuple, ast.List)):
        revisions: list[str] = []
        for element in value.elts:
            revisions.extend(_literal_revisions(element))
        return tuple(revisions)
    raise ValueError(f"unsupported revision literal: {ast.dump(value)}")


def parse_revision_file(path: str, content: str) -> Revision | None:
    tree = ast.parse(content, filename=path)
    revision: str | None = None
    down_revisions: tuple[str, ...] = ()

    for statement in tree.body:
        target_name: str | None = None
        value: ast.AST | None = None

        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target_name = statement.target.id
            value = statement.value
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    target_name = target.id
                    value = statement.value
                    break

        if target_name == "revision":
            parsed = _literal_revisions(value)
            if len(parsed) != 1:
                raise ValueError(f"{path}: revision must be one string literal")
            revision = parsed[0]
        elif target_name == "down_revision":
            down_revisions = _literal_revisions(value)

    if revision is None:
        return None
    return Revision(revision=revision, down_revisions=down_revisions, path=path)


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _ensure_commit_available(sha: str) -> None:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    _run_git(["fetch", "--no-tags", "--depth=1", "origin", sha])


def migration_files_for_ref(sha: str) -> list[tuple[str, str]]:
    _ensure_commit_available(sha)
    paths = _run_git(
        ["ls-tree", "-r", "--name-only", sha, "--", MIGRATION_VERSIONS_DIR]
    )

    migration_files: list[tuple[str, str]] = []
    for path in paths.splitlines():
        if not path.endswith(".py") or path.endswith("/__init__.py"):
            continue
        content = _run_git(["show", f"{sha}:{path}"])
        migration_files.append((path, content))

    if not migration_files:
        raise RuntimeError(
            f"No Alembic migration files found in {MIGRATION_VERSIONS_DIR} at {sha}"
        )
    return migration_files


def check_single_alembic_head(sha: str) -> tuple[bool, str]:
    revisions_by_id: dict[str, Revision] = {}
    duplicates: dict[str, list[str]] = {}

    for path, content in migration_files_for_ref(sha):
        revision = parse_revision_file(path, content)
        if revision is None:
            continue
        if revision.revision in revisions_by_id:
            duplicates.setdefault(
                revision.revision, [revisions_by_id[revision.revision].path]
            )
            duplicates[revision.revision].append(path)
        revisions_by_id[revision.revision] = revision

    if duplicates:
        duplicate_ids = ", ".join(sorted(duplicates))
        return False, f"Duplicate Alembic revision IDs: {duplicate_ids}"

    down_revision_ids = {
        down_revision
        for revision in revisions_by_id.values()
        for down_revision in revision.down_revisions
    }
    missing_down_revisions = sorted(down_revision_ids - set(revisions_by_id))
    if missing_down_revisions:
        return False, "Missing Alembic down revisions: " + ", ".join(
            missing_down_revisions
        )

    heads = sorted(set(revisions_by_id) - down_revision_ids)
    if len(heads) != 1:
        return False, f"Expected exactly one Alembic head, got: {', '.join(heads)}"
    return True, f"Single Alembic head: {heads[0]}"


def pr_touches_migrations(github: GitHub, number: int) -> bool:
    files = github.paginate(f"/repos/{github.repo}/pulls/{number}/files")
    return any(file.get("filename", "").startswith(MIGRATION_PREFIX) for file in files)


def resolve_branch_sha(github: GitHub, branch: str) -> str:
    quoted_branch = urllib.parse.quote(branch, safe="")
    branch_info = github.request(
        "GET", f"/repos/{github.repo}/branches/{quoted_branch}"
    )
    return branch_info["commit"]["sha"]


def head_contains_base(github: GitHub, base_sha: str, head_sha: str) -> bool:
    quoted_base = urllib.parse.quote(base_sha, safe="")
    quoted_head = urllib.parse.quote(head_sha, safe="")
    compare = github.request(
        "GET", f"/repos/{github.repo}/compare/{quoted_base}...{quoted_head}"
    )
    return compare.get("status") in {"ahead", "identical"}


def wait_for_refreshed_head(
    github: GitHub, pr_number: int, base_sha: str
) -> str | None:
    for _ in range(24):
        pr = github.request("GET", f"/repos/{github.repo}/pulls/{pr_number}")
        head_sha = pr["head"]["sha"]
        if head_contains_base(github, base_sha, head_sha):
            return head_sha
        time.sleep(5)
    return None


def latest_pr_head_sha(github: GitHub, number: int, fallback_sha: str) -> str:
    try:
        pr = github.request("GET", f"/repos/{github.repo}/pulls/{number}")
        return pr["head"]["sha"]
    except Exception as exc:
        print(
            f"PR #{number}: could not read latest head after failure: {exc}",
            file=sys.stderr,
        )
        return fallback_sha


def dispatch_workflows(github: GitHub, ref: str) -> None:
    for workflow in DISPATCH_WORKFLOWS:
        quoted_workflow = urllib.parse.quote(workflow, safe="")
        github.request(
            "POST",
            f"/repos/{github.repo}/actions/workflows/{quoted_workflow}/dispatches",
            {"ref": ref},
        )


def refresh_pr(github: GitHub, pr: dict[str, Any], base_sha: str) -> bool:
    number = pr["number"]
    head = pr["head"]
    old_sha = head["sha"]

    if not pr_touches_migrations(github, number):
        print(f"PR #{number}: no migration changes, skipping")
        return True

    head_repo = head.get("repo") or {}
    if head_repo.get("full_name") != github.repo:
        print(f"PR #{number}: fork PR, skipping branch update")
        return True

    print(f"PR #{number}: merging latest {BASE_BRANCH} into {head['ref']}")
    github.set_status(old_sha, "pending", f"Refreshing with latest {BASE_BRANCH}")

    try:
        github.request(
            "PUT",
            f"/repos/{github.repo}/pulls/{number}/update-branch",
            {"expected_head_sha": old_sha},
        )
    except GitHubError as exc:
        print(f"PR #{number}: update branch failed: {exc}", file=sys.stderr)
        github.set_status(old_sha, "failure", f"Could not merge latest {BASE_BRANCH}")
        return False

    try:
        new_sha = wait_for_refreshed_head(github, number, base_sha)
    except Exception as exc:
        message = f"Could not verify refreshed branch: {exc}"
        print(f"PR #{number}: {message}", file=sys.stderr)
        github.set_status(
            latest_pr_head_sha(github, number, old_sha), "failure", message
        )
        return False

    if new_sha is None:
        latest_pr = github.request("GET", f"/repos/{github.repo}/pulls/{number}")
        latest_sha = latest_pr["head"]["sha"]
        github.set_status(
            latest_sha, "failure", f"Branch did not include latest {BASE_BRANCH}"
        )
        return False

    github.set_status(new_sha, "pending", "Checking Alembic migration graph")
    try:
        ok, message = check_single_alembic_head(new_sha)
    except Exception as exc:
        message = f"Failed to validate Alembic graph: {exc}"
        print(f"PR #{number}: {message}", file=sys.stderr)
        github.set_status(new_sha, "failure", message)
        return False

    state = "success" if ok else "failure"
    if ok:
        try:
            dispatch_workflows(github, head["ref"])
        except Exception as exc:
            message = f"Failed to dispatch required workflows: {exc}"
            print(f"PR #{number}: {message}", file=sys.stderr)
            github.set_status(new_sha, "failure", message)
            return False
        if DISPATCH_WORKFLOWS:
            message = f"{message}; dispatched {', '.join(DISPATCH_WORKFLOWS)}"
    print(f"PR #{number}: {message}")
    github.set_status(new_sha, state, message)
    return ok


def main() -> int:
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    github = GitHub(token=token, repo=repo)
    base_sha = resolve_branch_sha(github, BASE_BRANCH)
    print(f"{BASE_BRANCH} is currently {base_sha}")

    prs = github.paginate(
        f"/repos/{repo}/pulls?state=open&base={urllib.parse.quote(BASE_BRANCH)}"
    )
    print(f"Found {len(prs)} open PR(s) targeting {BASE_BRANCH}")

    ok = True
    for pr in prs:
        ok = refresh_pr(github, pr, base_sha) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
