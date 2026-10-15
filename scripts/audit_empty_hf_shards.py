#!/usr/bin/env python3
"""Audit old Hugging Face Wikipedia gzip leaves; remove only confirmed empty orphans.

Dry-run by default. This does not remove repositories or any non-empty archives.
An empty archive referenced by build-manifest.json is reported, NOT silently
deleted: recovery metadata must be repaired before changing source coverage.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ORIGINAL = (
    "english spanish french german italian portuguese russian chinese norwegian "
    "simpleenglish"
).split()


def datasets(config: Path) -> list[str]:
    entries = json.loads(config.read_text(encoding="utf-8"))["languages"]
    # Previously excluded the October batch while it was not published.
    # Audits now cover every configured edition; unpublished projects are
    # safely reported as unavailable, never inferred to be empty.
    slugs = [x["slug"] for x in entries]
    slugs.extend(ORIGINAL)
    return sorted({f"ffatty/plaintext-wikipedia-full-{slug}" for slug in slugs}
                  - {"ffatty/plaintext-wikipedia-full-simpleenglish"})


def classify(sizes: dict[str, int], manifest_names: set[str] | None) -> dict:
    """Plan only exact zero-content candidates; absence is NOT proof of emptiness."""
    gzips = {name: int(size) for name, size in sizes.items()
             if name.startswith("data/") and name.endswith(".txt.gz")}
    suspicious = [name for name, size in gzips.items() if size <= 128]
    valid = [name for name, size in gzips.items() if size > 128]
    return {
        "files": len(gzips),
        "valid_size_archives": len(valid),
        "suspected_empty": sorted(suspicious),
        "referenced_candidates": sorted(x for x in suspicious
                                        if manifest_names is not None
                                        and x.removeprefix("data/") in manifest_names),
        "untracked_candidates": sorted(x for x in suspicious
                                        if manifest_names is not None
                                        and x.removeprefix("data/") not in manifest_names),
    }


def manifest_files(document: dict) -> set[str]:
    found = set()
    for root in document.get("roots", []):
        found.update(x for x in root.get("files", []) if isinstance(x, str))
        found.update(x.get("file") for x in root.get("successful_leaves", [])
                     if isinstance(x, dict) and isinstance(x.get("file"), str))
    return found


def audit_one(repo: str, apply: bool = False) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    result = {"repo": repo, "status": "unknown", "deleted": [], "alerts": []}
    try:
        entries = list(api.list_repo_tree(repo_id=repo, repo_type="dataset",
                                          path_in_repo="data", recursive=False,
                                          expand=True))
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code in (401, 403, 404):
            result["status"] = "not-accessible-or-unpublished"
            return result
        raise
    sizes = {x.path: x.size for x in entries
             if x.__class__.__name__ == "RepoFile" and x.size is not None}
    if "data/build-manifest.json" in sizes:
        path = hf_hub_download(repo_id=repo, repo_type="dataset",
                               filename="data/build-manifest.json",
                               token=os.environ.get("HF_TOKEN") or None)
        document = json.loads(Path(path).read_text("utf-8"))
        # Old manifest formats lacking per-root file references are not
        # sufficient evidence that an apparently orphaned file is disposable.
        referenced = (manifest_files(document)
                      if isinstance(document.get("roots"), list) and document["roots"]
                      else None)
    else:
        referenced = None  # Cannot establish safely that a candidate is orphaned.
    plan = classify(sizes, referenced)
    result["archive_count"] = plan["files"]
    if not plan["files"]:
        result["status"] = "no-published-archives"
        return result
    for name in plan["suspected_empty"]:
        path = hf_hub_download(repo_id=repo, repo_type="dataset",
                               filename=name, token=os.environ.get("HF_TOKEN") or None)
        try:
            with gzip.open(path, "rb") as src:
                has_article_bytes = bool(src.read(1))
        except (OSError, EOFError):
            has_article_bytes = False  # Invalid gzip; still require orphan provenance.
        if has_article_bytes:
            continue
        if referenced is None:
            result["alerts"].append(f"{name}: empty/invalid but no coverage manifest")
            continue
        if name.removeprefix("data/") in referenced:
            result["alerts"].append(f"{name}: empty/invalid AND referenced in coverage manifest")
            continue
        # Never delete the last archive; leave data recovery to a rebuild.
        if plan["valid_size_archives"] == 0:
            result["alerts"].append(f"{name}: last archive, needs rebuild not deletion")
            continue
        if apply:
            api.delete_file(repo_id=repo, repo_type="dataset",
                            path_in_repo=name,
                            commit_message=f"Remove verified-empty unreferenced archive {name}")
            result["deleted"].append(name)
        else:
            result["alerts"].append(f"{name}: safe orphan deletion candidate (dry run)")
    result["status"] = "audited"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("config/additional-languages.json"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", type=Path, default=Path("empty-shard-audit.json"))
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1 to 8")
    repos = datasets(args.config)
    reports = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(audit_one, repo, args.apply): repo for repo in repos}
        for future in as_completed(futures):
            repo = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"repo": repo, "status": "error",
                          "error": f"{type(exc).__name__}: {exc}"}
            reports.append(result)
            if result.get("deleted") or result.get("alerts") or result["status"] == "error":
                print(json.dumps(result, ensure_ascii=False), flush=True)
    reports.sort(key=lambda x: x["repo"])
    summary = {
        "total_repositories": len(reports),
        "audited": sum(x["status"] == "audited" for x in reports),
        "missing_or_unavailable": sum(x["status"] == "not-accessible-or-unpublished"
                                      for x in reports),
        "no_archives": sum(x["status"] == "no-published-archives" for x in reports),
        "errors": sum(x["status"] == "error" for x in reports),
        "deleted_files": sum(len(x.get("deleted", [])) for x in reports),
        "alert_count": sum(len(x.get("alerts", [])) for x in reports),
        "results": reports,
    }
    args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n",
                           encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "results"}))
    if summary["errors"]:
        raise SystemExit("Some repository audits failed; inspect the published report")


if __name__ == "__main__":
    main()
