#!/usr/bin/env python3
"""Read-only publishing credentials preflight; never uploads data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from huggingface_hub import HfApi


def check(kaggle: bool = True, zenodo: bool = True) -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing")
    HfApi(token=token).whoami()
    print("Hugging Face authentication OK", flush=True)

    if kaggle:
        if not os.getenv("KAGGLE_TOKEN"):
            raise RuntimeError("KAGGLE_TOKEN is missing")
        subprocess.run([sys.executable, "scripts/kaggle_auth.py", "--owner", "ffatty"],
                       check=True)
        print("Kaggle CLI authentication OK", flush=True)

    if zenodo:
        import requests
        token = os.getenv("ZENODO_TOKEN")
        if not token:
            raise RuntimeError("ZENODO_TOKEN is missing")
        response = requests.get(
            "https://zenodo.org/api/deposit/depositions",
            params={"size": 1},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        response.raise_for_status()
        if not isinstance(response.json(), list):
            raise RuntimeError("Unexpected Zenodo response")
        print("Zenodo read access OK (write scope is not guaranteed)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-kaggle", action="store_true")
    parser.add_argument("--no-zenodo", action="store_true")
    args = parser.parse_args()
    check(kaggle=not args.no_kaggle, zenodo=not args.no_zenodo)


if __name__ == "__main__":
    main()
