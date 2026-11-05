# Releasing Wikipedia datasets: HF → Kaggle → Zenodo

This project publishes **updates to existing dataset IDs**, not new datasets per format.
Release workflows are **manual-only**. The regular `test.yml` workflow runs
offline checks automatically; it does not read secrets or publish data.

## Three destinations

| Destination | Contents | Trigger |
|---|---|---|
| Hugging Face | Existing `.txt.gz` + independently checked Parquet + SQLite | Manual `structured-fleet.yml` |
| Kaggle | New version of **existing** dataset, keeping original files; Parquet + SQLite when staging is safe | Same manual fleet job after HF |
| Zenodo | **Parquet only**, compact source provenance, persistent DOI | Same manual fleet run (`publish_zenodo=true` by default); separate `zenodo-parquet.yml` also defaults to publish |

The **source-of-truth** is the published HF `data/*.txt.gz` archive.
Structured exports do not require a new Wikimedia XML download.

## First: credentials, no publication

GitHub Actions → **Check publishing credentials (read-only)** → Run workflow.

Tests: `HF_TOKEN` authenticated identity/dataset visibility; `KAGGLE_TOKEN`
with the real Kaggle CLI against an existing protected dataset; `ZENODO_TOKEN`
read-only deposition listing. Credentials are never written to logs.

A successful read check **does not guarantee write scopes**, quotas or publication
approval. The Kaggle CI environment must install `six` alongside `kaggle`,
as the current client can import `six` without bringing that package in.

## Then: the multilingual batch

GitHub Actions → **Rebuild structured editions across all published Wikipedias**.

Defaults: **publish all three destinations**, Parquet + SQLite on HF/Kaggle and Parquet only on Zenodo, maximum **2500 MiB**
compressed staging per Kaggle dataset. The workflow discovers existing published
HF language repositories; each language runs in an independent matrix job
(`max-parallel: 3`). Within a language:

1. Inspect source LFS hashes and existing format manifests.
2. Skip current, checksum-matched formats; regenerate only stale/missing formats.
3. Publish structured artifacts first, provenance manifests last, without touching
   existing raw source files.
4. After verifying all three publisher credentials before conversion, stage **the entire existing Kaggle version**
   before adding the verified formats. Skip an unchanged version entirely.
5. Publish or version the matching Zenodo DOI automatically, with Parquet only.
6. **Fail visibly** if a required Kaggle update exceeds the staging budget,
   runner disk or file count; HF and Zenodo publishing remain independent.
   No 'successful' three-destination release can hide skipped uploads.

**Large languages:** Kaggle full-version uploads can exceed a standard Actions
runner's disk. A safe Kaggle skip is expected, not a false success or an
invitation to discard raw files. A larger runner or separate external worker is
needed for releases above the configured staging budget.

**Reruns:** Manifests preserve source SHA256 and each artifact's SHA256. A rerun
can skip successful formats. Single-language jobs can be retried without
rebuilding the entire corpus, provided source archives have not changed.

## Zenodo (research archive)

GitHub Actions → **Zenodo Parquet research release**. With the default `publish=true`, the dedicated Zenodo workflow performs a
Parquet-only release. Set `publish=false` to run checks without uploading.

The publisher searches your existing Zenodo depositions before creating a
record; unchanged manifest fingerprints are skipped and changed ones create
a new version. You can still supply `existing_record_id` to select a specific
record. Review the quota and fair-use expectations for very large collections. Verify the token has `deposit:write` and
`deposit:actions` scopes in addition to read access. Zenodo's default
per-record limit is 50 GB, with at most 100 files. This exporter uploads
**Parquet + one provenance.json**, never SQLite or original gzip.

Zenodo publication is immutable and DOI minting cannot be undone.

## Historical failure audit (October–November 2026)

| Observed | Root cause / correction |
|---|---|
| October 15: 30+ backfill jobs all failed | Same rejected Kaggle credentials; check the credential **once** before releasing a matrix. |
| October 14: multi-edition plans failed immediately | Direct script invocation lacked package imports; test `python scripts/... --help` in CI. |
| October 14: dataset card publishing failed | Incorrect expected archive counts or undefined `expected`; derive **published** filenames from HF and **root coverage** from the manifest. |
| November 1: English refresh downloaded/converted a shard then failed | Kaggle rejected credentials after the expensive conversion. The refresh now publishes HF first and checks Kaggle CLI before downloading Kaggle's entire corpus. |
| November 5: structured Simple English exported to HF but Kaggle failed | HF and Kaggle have independent status; subsequent CI should not rebuild current HF results or conceal Kaggle failures. |
| November 5: regression suite could not start Kaggle CLI | Missing `six` (undeclared/transitively missing dependency); install explicitly. |

The scheduled raw refresh workflows **defer Kaggle versioning** when pre-existing
Parquet/SQLite would become stale. Run the fleet to regenerate the structured
formats, then version a coherent Kaggle dataset.

## Data contract

Parquet columns: `title`, `text`, `source_file`, `source_ordinal`.
The ordinal is one-based within its gzip input; **it is not a Wikipedia page ID**.
Some source recoveries are partial. Check the `data/build-manifest.json` coverage
flags before claiming complete corpora. Article content is from Wikimedia
contributors (CC BY-SA 4.0); converter source is in this repository (MIT).
Attribution obligations still apply.

## Local offline checks

```sh
python -m pip install -r REQUIREMENTS.TXT pyarrow zstandard huggingface_hub kaggle six pyyaml requests
python -m compileall -q scripts tests
python -m unittest discover -s tests -v
```

## GitHub Actions storage: why the old builds broke

The older XML rebuilds passed multi-GB `.txt.gz` leaves between jobs using
`actions/upload-artifact`. Those **persisted Actions artifacts** are billed
against the account's storage quota (potentially as little as 500 MB). Deleting
temporary files from an ephemeral runner does **not** delete already uploaded
artifacts. The new structured fleet does **not upload any Actions artifacts**:
it reads source archives from HF and uploads finished releases to HF, Kaggle,
and Zenodo instead.

`cleanup-wiki-artifacts.yml` runs after legacy rebuild workflows complete and
deletes only matching `full-...wiki-N` and `recovered-...wiki-N` intermediate
artifacts. Temporary files are also scoped to per-shard directories and
removed on exit. Historical usage may take time to update in GitHub billing.

On the standard runner, Kaggle may exceed disk capacity for the largest
editions. A size/disk preflight now reports this as a **failed Kaggle release**;
not an invisible skip. Set `kaggle_stage_mb` appropriately on a runner with
sufficient capacity. Large editions still require a larger runner or another
execution environment—this is distinct from the Actions artifact quota.
