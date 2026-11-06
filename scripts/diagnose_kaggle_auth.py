#!/usr/bin/env python3
"""Read-only, isolated diagnostics for Kaggle Actions credentials.

Checks actual CLI account authorization plus Kaggle's protected REST endpoint.
Never print credentials, subprocess output, or HTTP bodies.
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from scripts.kaggle_auth import candidates

PROTECTED_STATUS = (
    "https://www.kaggle.com/api/v1/datasets/status/"
    "ffatty/plain-text-wikipedia-simpleenglish"
)


def check_rest(authorization: str) -> str:
    request = urllib.request.Request(PROTECTED_STATUS, headers={
        "Authorization": authorization, "User-Agent": "PlainTextWikipedia-CI-auth-diagnostic/1.0",
    })
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}"
    except (TimeoutError, OSError, urllib.error.URLError):
        return "network error"


def check_cli(auth: dict[str, str]) -> tuple[int, str]:
    # Forbid accidentally using a cached OAuth login on this runner.
    with tempfile.TemporaryDirectory(prefix="kaggle-auth-verify-") as tmp:
        home = Path(tmp)
        config = home / ".kaggle"
        config.mkdir()
        env = {key: value for key, value in os.environ.items()
               if key not in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY",
                              "KAGGLE_TOKEN", "KAGGLE_CONFIG_DIR")}
        env.update(auth)
        env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"),
                   KAGGLE_CONFIG_DIR=str(config))
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "quota", "--format", "json"],
            capture_output=True, text=True, env=env, timeout=40, check=False,
        )
        # Do not emit raw CLI stdout/stderr: some CLIs echo credential values.
        diagnostics = (result.stdout + "\n" + result.stderr).lower()
        if result.returncode == 0:
            return 0, "authorized"
        if "authentication required" in diagnostics or "unauthenticated" in diagnostics:
            return result.returncode, "authentication required"
        if "401" in diagnostics:
            return result.returncode, "HTTP 401"
        if "403" in diagnostics:
            return result.returncode, "HTTP 403"
        if "404" in diagnostics:
            return result.returncode, "HTTP 404"
        return result.returncode, "other CLI error"


def diagnose(secret: str, owner: str) -> list[tuple[str, str, str]]:
    results = []
    for mode, auth in candidates(secret, owner):
        authorization = ("Bearer " + auth["KAGGLE_API_TOKEN"]
                         if "KAGGLE_API_TOKEN" in auth else "Basic " +
                         base64.b64encode((auth["KAGGLE_USERNAME"] + ":" +
                                           auth["KAGGLE_KEY"]).encode()).decode())
        code, cli_result = check_cli(auth)
        rest_result = check_rest(authorization)
        results.append((mode, cli_result, rest_result))
    return results


def main() -> None:
    secret = os.getenv("KAGGLE_TOKEN", "")
    if not secret:
        print("KAGGLE_TOKEN missing")
        raise SystemExit(1)
    # Report format categories, not secret bytes or secret-derived substrings.
    cleaned = secret.strip()
    print("Secret format:",
          "legacy-json" if cleaned.startswith("{") else
          "new-API-token" if cleaned.startswith("KGAT_") else
          "bearer-prefixed" if cleaned.lower().startswith("bearer ") else
          "multiline" if len(cleaned.splitlines()) > 1 else
          "opaque-single-line")
    print("Has internal whitespace:", any(c.isspace() for c in cleaned))
    results = diagnose(secret, "ffatty")
    for mode, cli, rest in results:
        print(f"{mode}: isolated CLI quota={cli}; authenticated REST={rest}")
    if not any(cli == "authorized" for _, cli, _ in results):
        raise SystemExit("No credential mode was authorized in a clean runner")


if __name__ == "__main__":
    main()
