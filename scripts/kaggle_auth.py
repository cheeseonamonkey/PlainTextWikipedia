#!/usr/bin/env python3
"""Kaggle CLI authentication for both legacy and modern GitHub Actions secrets.

Accept KAGGLE_TOKEN containing:
- {"username":"...","key":"..."} (legacy kaggle.json)
- a new Kaggle access token (KAGGLE_API_TOKEN)
- a legacy API key string with the owner derived from the intended dataset
A key string is tried in both modes because secret names do not identify
credential types reliably. NEVER print credentials or token-derived substrings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def candidates(secret: str, owner: str) -> list[tuple[str, dict[str, str]]]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
        raise ValueError("Invalid Kaggle owner")
    value = secret.strip()
    if not value:
        raise ValueError("KAGGLE_TOKEN is missing")
    if value.startswith("{"):
        item = json.loads(value)
        if not isinstance(item, dict):
            raise ValueError("KAGGLE_TOKEN JSON must be an object")
        if item.get("username") and item.get("key"):
            return [("legacy-json", {"KAGGLE_USERNAME": str(item["username"]),
                                     "KAGGLE_KEY": str(item["key"])})]
        if item.get("token"):
            return [("access-token", {"KAGGLE_API_TOKEN": str(item["token"])})]
        raise ValueError("JSON must contain username/key or token")
    # New token first, then the old API key format. Neither is printed.
    return [
        ("access-token", {"KAGGLE_API_TOKEN": value}),
        ("legacy-key", {"KAGGLE_USERNAME": owner, "KAGGLE_KEY": value}),
    ]


def persist_verified_credentials_to_actions(auth: dict[str, str],
                                            env_file: str | None) -> None:
    """Make credentials verified in this process available to later CI steps.

    GitHub Actions starts every shell step as a NEW process. Authenticating
    with environment variables in a child process is not persistent; nor is
    writing kaggle.json reliable when the subsequent CLI prioritizes env vars.
    GITHUB_ENV is GitHub's secure, intended cross-step environment channel.
    Nothing secret is written to logs, outputs, commits, or job summaries.
    """
    if not env_file:
        return
    allowed = {"KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"}
    if not auth or set(auth) - allowed:
        raise ValueError("Unexpected Kaggle authentication environment keys")
    with open(env_file, "a", encoding="utf-8") as stream:
        for key, value in auth.items():
            if not value or "\n" in value or "\r" in value:
                raise ValueError("Invalid Kaggle credential for GitHub Actions environment")
            stream.write(f"{key}={value}\n")


def verify_and_configure(secret: str, owner: str, home: Path | None = None) -> str:
    home = home or Path.home()
    folder = home / ".kaggle"
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    folder.chmod(0o700)
    config, token_path = folder / "kaggle.json", folder / "access_token"
    for path in (config, token_path):
        path.unlink(missing_ok=True)
    modes = candidates(secret, owner)
    for mode, auth in modes:
        clean = {key: value for key, value in os.environ.items()
                 if key not in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME",
                                "KAGGLE_KEY", "KAGGLE_TOKEN", "KAGGLE_CONFIG_DIR")}
        clean.update(auth)
        # This explicitly requires account-authenticated listings, not a public
        # dataset lookup that could succeed without any credentials.
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "list", "--mine"],
            env=clean, capture_output=True, text=True, timeout=90, check=False,
        )
        if result.returncode != 0:
            print(f"::warning::Kaggle authentication mode {mode} rejected",
                  file=sys.stderr, flush=True)
            continue
        if "KAGGLE_API_TOKEN" in auth:
            token_path.write_text(auth["KAGGLE_API_TOKEN"], encoding="utf-8")
            token_path.chmod(0o600)
        else:
            config.write_text(json.dumps({"username": auth["KAGGLE_USERNAME"],
                                          "key": auth["KAGGLE_KEY"]}),
                              encoding="utf-8")
            config.chmod(0o600)
        # The authenticated subprocess had explicit environment credentials;
        # preserve that exact successful mode for all FOLLOWING Actions steps.
        persist_verified_credentials_to_actions(auth, os.environ.get("GITHUB_ENV"))
        print(f"Kaggle authenticated using {mode}", flush=True)
        return mode
    raise RuntimeError("No Kaggle credential mode authenticated. Rotate/update the "
                       "KAGGLE_TOKEN GitHub Actions secret at Kaggle Settings > API; "
                       "provide either a modern API token or legacy kaggle.json.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="ffatty")
    args = parser.parse_args()
    verify_and_configure(os.environ.get("KAGGLE_TOKEN", ""), args.owner)


if __name__ == "__main__":
    main()
