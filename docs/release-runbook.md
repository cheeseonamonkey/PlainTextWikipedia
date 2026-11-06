# Releasing Wikipedia datasets: HF → Kaggle → Zenodo

This project publishes **updates to existing dataset IDs**, not new datasets per format.
Release workflows are **manual-only**. The regular `test.yml` workflow runs
offline checks automatically; it does not read secrets or publish data.

## Uniform Wikimedia rebuilds

Every Wikimedia edition uses `.github/workflows/rebuild-full.yml` for source
selection, download, adaptive conversion, coverage validation, and publishing.
The `language` project code (for example `simplewiki`, `enwiki`, or
`frwiki`) determines source URLs and existing dataset destinations. Size and
source layout determine processing concurrency. Simple English retains its
legacy single gzip archive and dataset IDs; larger editions may publish up to
16 archives. `rebuild-simple.yml`, `refresh-simple.yml`, and
`refresh-full-monthly.yml` are entrypoint/schedule wrappers, not independent
pipelines. A monthly full English refresh now rebuilds the complete edition,
not a rotating single shard.

The corpus feature release is separate: after source publication, run
`article-features-fleet.yml` to regenerate the 200D embeddings. Feature
fusion policy is versioned; changing its weights does **not** mutate already
published feature files until their feature job runs successfully.

## Three destinations

| Destination | Contents | Trigger |
|---|---|---|
| Hugging Face | Existing `.txt.gz` + independently checked Parquet | Manual `structured-fleet.yml` |
| Kaggle | New version of **existing** dataset, keeping original files; Parquet when staging is safe | Same manual fleet job after HF |
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

GitHub Actions → **Publish all Wikipedia editions (original + Parquet)**.

Defaults: **publish all three destinations**, original gzip + Parquet on HF/Kaggle,
Parquet only on Zenodo, **8 concurrent languages**, and **automatic staging limits**
based on available runner disk (no arbitrary 5 GiB cap).

The controller discovers existing Hugging Face language repositories, checks
the publisher credentials **once**, then calls the same reusable per-language
workflow available from **Publish one Wikipedia dataset**. GitHub queues the
remaining matrix jobs; no separate workflow-dispatch events or artifact handoffs.
Use `parallelism` (1–12) to tune concurrency, or `languages` (comma-separated
**Wikimedia codes**, project IDs, dataset slugs or repo IDs) to retry only failed
editions. For example, `simple,en` selects Simple English and Full English;
`shwiki` selects Serbo-Croatian. Codes are resolved against existing published
repos using `config/additional-languages.json` and `config/datasets.env`; unknown
or ambiguous selections fail before any release jobs start.

The cheap Simple English edition runs first as a smoke check, followed by the
largest likely editions to avoid a long trailing job. Actual runtime depends
on Wiki shard sizes, host rate limits, runner capacity and upload speeds.

Within each language:

1. Compare source LFS digests with published Parquet manifests; skip current shards.
2. Stream outdated gzip shards into larger Parquet row groups using faster
   Zstandard compression. Avoid a duplicate full-source checksum pass.
3. **Atomically commit** Parquet and its SHA256 provenance manifest on HF;
   retain the original plaintext files.
4. Check Kaggle's small remote manifest and filename inventory **before**
   downloading any full dataset. Skip unchanged versions. Otherwise stage
   all existing files, remove obsolete SQLite and publish a coherent new version.
5. Automatically publish or version the matching Zenodo DOI (Parquet only).
   An unchanged verified release skips re-uploading.
6. **Fail visibly** on required Kaggle disk/file/remote failures, while allowing
   HF and Zenodo to finish. Each language reports the outcome for all three.

**Large languages:** Kaggle full-version uploads can exceed a standard Actions
runner's disk. Large editions fail explicitly when staging would exhaust actual runner
disk. A larger runner or external worker may be needed; raw files are never
discarded to fake a successful Kaggle version.

**Performance:** Kaggle and Zenodo reuse downloaded HF Parquet blobs
within the same runner instead of each downloading an identical artifact.
Parquet conversion caps working batches at 16,384 articles **and 32 MiB**
of decoded text to avoid oversized-memory row groups. New Parquet manifests
record SHA256 plus a Zenodo-compatible MD5 from a single file scan; matching
draft files can be skipped without downloading. Per-language Actions job
summaries report wall time by publisher, so optimize based on real bottlenecks
after the `simple,en` test rather than estimated percentages.

**Reruns:** Source/artifact SHA256 manifests allow successful HF shards,
unchanged Kaggle datasets, and unchanged Zenodo records to skip redundant work.
Use `languages` to retry only failures (e.g. `en,sh`).
Only temporary network errors are retried with exponential backoff and jitter;
authentication and bad-source errors fail immediately.

**Capacity:** GitHub's matrix limit is **256 language jobs per run**. The
current fleet fits under that limit. A future larger fleet requires sharding
the controller into multiple batches; discovery fails explicitly rather than
silently dropping language editions.

## Representation policy

The publishing pipeline now produces **only original gzip and Parquet**.
No SQLite or FTS5 databases are generated by the release workflows. Older SQLite outputs in Hugging Face remain available for compatibility.
New Kaggle versions exclude generated SQLite artifacts; older Kaggle
versions remain downloadable. Neither destination regenerates SQLite. This eliminates a second
conversion, compression and upload per source shard, and reduces Kaggle's
per-dataset file-count pressure. The standalone legacy exporter still contains
SQLite support for reproducibility and existing tests; the publisher does not use it.

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

The scheduled raw refresh workflows **defer Kaggle versioning** when existing
structured files would become stale. Run the fleet to refresh the missing Parquet outputs, then version a
coherent Kaggle dataset.

## Data contract

Parquet v2 columns: `title`, `text`, `source_file`, `source_ordinal`, `char_count`, `byte_count`, `word_count`, `article_url`. The text counts are per-article; word count is whitespace-based, not multilingual tokenization. Article links are best-effort (not canonical or validated). The first post-upgrade export rebuilds v1 Parquet once.
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
artifacts. The new original-plus-Parquet fleet does **not upload any Actions artifacts**:
it reads source archives from HF and uploads finished releases to HF, Kaggle,
and Zenodo instead.

`cleanup-wiki-artifacts.yml` runs after legacy rebuild workflows complete and
deletes only matching `full-...wiki-N` and `recovered-...wiki-N` intermediate
artifacts. Temporary files are also scoped to per-shard directories and
removed on exit. Historical usage may take time to update in GitHub billing.

On the standard runner, Kaggle may exceed disk capacity for the largest
editions. A dynamic disk-space preflight reports this as a **failed Kaggle
release**, not an invisible skip. Set `kaggle_stage_mb` above zero only when
you intentionally want a stricter cap. Large editions still require a larger runner or another
execution environment—this is distinct from the Actions artifact quota.
