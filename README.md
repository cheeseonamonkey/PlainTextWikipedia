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
- **Rebuild all English Wikipedia shards** is manual-only and uses one standard runner per shard, then publishes the complete set to Kaggle.

Each full-English shard is independently compressed. Hugging Face receives the changed shard directly. Kaggle versions are complete dataset publishes; the monthly workflow downloads the existing Kaggle version, replaces one shard, and uploads a new version.

Add these repository secrets before publishing:

- `KAGGLE_JSON`: the contents of the Kaggle API JSON credential
- `HF_TOKEN`: a Hugging Face token with write access to the dataset repositories

Workflows validate that output is non-empty and gzip-readable before publishing. Publishing steps are skipped when the corresponding secret is absent.

## Data sources and license

- Simplified English Wikipedia dumps: https://dumps.wikimedia.org/simplewiki/
- English Wikipedia page-range dumps: https://dumps.wikimedia.org/enwiki/

Wikipedia text is available under the [Creative Commons Attribution-ShareAlike license (CC-BY-SA)](https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content). The converter code is MIT licensed; that does not change the license of converted Wikipedia text.
