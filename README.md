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

The output files are independent `.txt.gz` shards, assigned by stable page ID. The converter also accepts uncompressed XML, `.gz`, and `.xz` inputs.

## GitHub Actions

- **Refresh Simple English dataset** runs quarterly and can also be started manually.
- **Rebuild complete Simple English dataset** is manual-only for a clean rebuild or recovery.

Add these repository secrets before publishing:

- `KAGGLE_JSON`: the contents of the Kaggle API JSON credential
- `HF_TOKEN`: a Hugging Face token with write access to the dataset repository

The workflows validate that output is non-empty and gzip-readable before publishing. Publishing steps are skipped when the corresponding secret is absent.

## Data sources and license

- Simplified English Wikipedia dumps: https://dumps.wikimedia.org/simplewiki/
- English Wikipedia dumps: https://dumps.wikimedia.org/enwiki/

Wikipedia text is available under the [Creative Commons Attribution-ShareAlike license (CC-BY-SA)](https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content). The converter code is MIT licensed; that does not change the license of converted Wikipedia text.
