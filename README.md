# PlainTextWikipedia

Convert current Wikimedia XML dumps into compressed plaintext shards.

The converter keeps one current revision per main-namespace page, skips redirects, removes MediaWiki markup, and writes each article as:

```text
Article title
Plain article text

```

## Local use

Install the dependencies and run:

```bash
python -m pip install -r REQUIREMENTS.TXT
python wiki_to_text.py dump.xml.bz2 output --shards 10 --prefix enwiki
```

The converter accepts one or more independently parseable XML, `.bz2`, `.gz`, or `.xz` inputs. Page-range files can therefore be combined into one logical shard without downloading a whole language dump.

## GitHub Actions

All dataset-specific knobs live in [config/datasets.env](config/datasets.env):

- source URLs and Wikimedia language
- logical shard count
- Kaggle dataset IDs
- Hugging Face dataset repositories

Change that file when a dataset moves or the shard layout changes. The schedule expressions are at the top of each workflow so cadence changes remain visible.

- **Refresh Simple English dataset** runs quarterly and can also be started manually.
- **Rebuild complete Simple English dataset** is manual-only.
- **Refresh one English Wikipedia shard** runs monthly. Month 1 refreshes shard 1, month 2 shard 2, and so on; a manual run can select a shard.
- **Rebuild Wikipedia shards** is manual-only and accepts any Wikimedia project code (`enwiki`, `nowiki`, `frwiki`, `dewiki`, …), one standard runner per shard, then publishes the complete set to Kaggle.

Each full shard is independently compressed. Hugging Face receives the changed shard directly. Kaggle versions are complete dataset publishes; the monthly workflow downloads the existing Kaggle version, replaces one shard, and uploads a new version.

Add these repository secrets before publishing:

- `KAGGLE_TOKEN`: a Kaggle API token (`KAGGLE_API_TOKEN`)
- `HF_TOKEN`: a Hugging Face token with write access to the dataset repositories

Workflows validate that output is non-empty and gzip-readable before publishing. Publishing steps are skipped when the corresponding secret is absent.

## Provenance and credits

Published Kaggle descriptions and Hugging Face dataset cards credit the projects that produced the data:

1. [cheeseonamonkey/PlainTextWikipedia](https://github.com/cheeseonamonkey/PlainTextWikipedia) — maintained fork that generates and publishes these datasets.
2. [daveshap/PlainTextWikipedia](https://github.com/daveshap/PlainTextWikipedia) — original upstream project (unmaintained).

The credit text is configured once in [config/datasets.env](config/datasets.env) (`SOURCE_REPO_URL`, `UPSTREAM_REPO_URL`) and rendered into [kaggle-metadata.json](kaggle-metadata.json), [kaggle-metadata-full.json](kaggle-metadata-full.json), and the Hugging Face card ([config/hf-dataset-card.md](config/hf-dataset-card.md)).

Kaggle has no dedicated provenance field for datasets (`provenanceSources` exists only for models), so the credit is carried by the dataset **subtitle** (`Built by cheeseonamonkey/PlainTextWikipedia`, 20–80 characters) and the opening sentence of the markdown **description**, which links the repository name inline. The Hugging Face card links it in the same way, as a hyperlinked word in the intro sentence.

## Data sources and license

- Simplified English Wikipedia dumps: https://dumps.wikimedia.org/simplewiki/
- Any Wikimedia project page-range dumps: https://dumps.wikimedia.org/ (for example https://dumps.wikimedia.org/enwiki/, https://dumps.wikimedia.org/nowiki/)

Wikipedia text is available under the [Creative Commons Attribution-ShareAlike license (CC-BY-SA)](https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content). The converter code is MIT licensed; that does not change the license of converted Wikipedia text.
