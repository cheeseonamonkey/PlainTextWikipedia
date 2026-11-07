# Publication refactor review

This describes the proposed implementation in PR #40. Production runs remain
paused; the PR is unmerged and a full English build has not been launched.

| Drift/failure found | Correction |
|---|---|
| Separate 5K TF-IDF and unrestricted fastText vocabularies | One aggressively filtered/stemmed top-10K vocabulary for both, plus BM25 |
| Unsupported languages silently depended on optional packages | Required normalization dependencies; record unsupported morphology/stopword coverage explicitly |
| Inflections could inflate document frequency | Merge normalized forms within each article before updating corpus DF |
| Redundant coverage columns after vocabulary consolidation | Remove `tfidf_coverage`; schema v7 has 32 columns |
| Bekko input was case-folded with lexical tokens | Feed an original readable excerpt; pin the model revision |
| Backend-specific dependency guidance was confused with the generic model quickstart | Use Bekko's documented OpenVINO dependency ranges; install CPU PyTorch explicitly |
| Any upload error prevented later publishers from trying | Attempt all three, preserve errors and fail the overall job when any target fails |
| Workflow duplicated a language-name switch and repository resolver | Use one catalog-backed resolver; validate language and destination agreement |
| Zenodo success was inferred from upload/action responses | Verify the submitted record’s exact file set, sizes and checksums |
| Test and build dependencies were separately maintained | Share feature requirements; keep encoder-only dependencies separate |
| Small Git-stored HF sources were rejected for lacking an LFS digest | Verify their Git blob identity instead; preserve SHA256 in provenance |
| Release metadata version/dimensions were repeated across modules | Central schema/version and encoder constants |
| Embedding logging printed a line for every 16 articles | Report progress every 1024 articles and at completion |

Raw article text and raw word-count dictionaries remain complete. Feature
filtering limits vocabulary, not article coverage. Raw and reduced corpus word
statistics remain available, including terms outside the top 10K.

## Review coverage

Reviewed the active workflows and their script entrypoints, source validation,
article ingestion and corpus aggregation, vocabulary/training/vector composition,
Parquet schema, bundle verification/packaging, publication retry paths, dataset
links, dependency declarations and user-facing documentation. Historical retired
publishing helpers still exist for old tests/tools; active workflows cannot invoke
their former release commands.

## Evidence and remaining validation

142 offline tests pass, and the OpenVINO requirements resolve without installing
models. Workflow lint and Python compilation pass. Tests exercise real fastText training, multilingual stemming and Unicode,
normalized corpus DF, shared training vocabulary, unchanged article/text coverage,
checksum/schema rejection, publisher failure aggregation, retry behavior and
catalog mappings. Bekko inference is substituted in offline corpus fixtures.

The next corpus acceptance test is **full English, build-only**, using the same
pipeline and full source coverage. Disk/time capacity, live Bekko throughput and
actual publisher acceptance remain unmeasured. Local tests are not evidence that
English fits the standard GitHub runner. The model's language coverage is broad
but does not establish equal quality across every Wikipedia edition.

References: [Bekko OpenVINO setup](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m#optimized-inference),
[PyStemmer](https://github.com/snowballstem/pystemmer),
[output contract](article-features.md).
