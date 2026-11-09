# PlainTextWikipedia

Convert Wikipedia dumps into full-text, article-level datasets with one shared
pipeline for all languages. Source content is preserved; feature selection never
removes articles.

## Output

Each completed edition produces **the same five files** for Hugging Face, Kaggle,
and Zenodo:

| File | Contents |
|---|---|
| `articles.parquet` | One article per row; original text, lexical metrics, TF-IDF/BM25, fastText and Bekko vectors |
| `articles.txt.xz` | Entire corpus as title/body/blank-line records |
| `features.zip` | Corpus profile, selected vocabulary, and complete word statistics |
| `models.zip` | Compressed fastText model and feature configuration |
| `manifest.json` | Source provenance, coverage, links, checksums, and output versions |

See [schema and release contract](docs/article-features.md) for the exact 32-column
schema, language handling, verification, and limitations.

## Build and publish

- **Existing corpus:** Run [Build complete Wikipedia bundle](.github/workflows/article-features-one.yml)
  with the existing Hugging Face dataset ID (e.g. `ffatty/plaintext-wikipedia-full-english`).
- **Multiple editions:** Run [Wikipedia bundle fleet](.github/workflows/article-features-fleet.yml).
  Blank language selection includes all existing eligible datasets.
- **Fresh Wikimedia dump:** Use [Rebuild Wikipedia shards](.github/workflows/rebuild-full.yml)
  or [Build all Wikipedia languages](.github/workflows/rebuild-languages.yml).
  Both feed the same final bundle builder.

Every manual build defaults to `publish: false`, including fresh-source rebuilds.
Build-only runs verify locally and retain the bundle temporarily. When explicitly
enabled, publishing requires all three credentials, builds **everything first**,
then uploads identical artifacts to Hugging Face, Kaggle and Zenodo. Individual
publishers may fail; a failed multi-service release is reported as incomplete.

Before a large run, CI checks workflow syntax, source-coverage invariants,
bundle integration and publisher failure behavior. The bundle workflow also tests
real CPU Bekko inference **before** processing the corpus. No full English
runtime, disk requirement or cross-service publishing success has been established
by these small tests.

## Development

```bash
python -m pip install -r REQUIREMENTS.TXT -r requirements-features.txt
python -m unittest discover -s tests -v
```

CI installs its CPU dependencies from the checked-in wheelhouse and loads the
checked-in Bekko model. See [vendored dependencies](vendor/README.md) for
platform, setup, licensing and update details. The configured dataset catalog is in
`config/`; existing scripts and tests remain in `scripts/` and `tests/`.

## Attribution and licensing

- [Wikimedia dumps](https://dumps.wikimedia.org/) — Wikipedia content and
  applicable Creative Commons Attribution-ShareAlike requirements.
- [This maintained fork](https://github.com/cheeseonamonkey/PlainTextWikipedia).
- [Original project](https://github.com/daveshap/PlainTextWikipedia).

The code license does not replace the license for Wikipedia article content.
