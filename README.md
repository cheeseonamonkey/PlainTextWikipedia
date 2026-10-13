# PlainTextWikipedia

Convert official Wikimedia dumps into independently compressed, UTF-8 plaintext article archives. This maintained fork automates multilingual builds and publishes the output to **Hugging Face** and **Kaggle**.

## Published datasets and destinations

| Wikipedia edition | Hugging Face | Kaggle |
|---|---|---|
| Simple English (`simplewiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plain-text-wikipedia-simpleenglish) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish) |
| English (`enwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-english) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-english) |
| Norwegian (`nowiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-norwegian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-norwegian) |
| Spanish (`eswiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-spanish) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-spanish) |
| French (`frwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-french) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-french) |
| German (`dewiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-german) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-german) |
| Italian (`itwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-italian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-italian) |
| Portuguese (`ptwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-portuguese) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-portuguese) |
| Russian (`ruwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-russian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-russian) |
| Chinese (`zhwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-chinese) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-chinese) |

Links above are the **configured publishing destinations**. A dataset may be pending or incomplete if its latest workflow is still running or a publishing service rejected an upload. Check the dataset's latest files and `data/build-manifest.json` (on Hugging Face) or `build-manifest.json` (on Kaggle) for verified coverage, available archives, and missing ranges.

## Dataset format

Each `.txt.gz` file is a standalone gzip-compressed UTF-8 text archive with a title, one cleaned article body, then a blank line for each article:

```text
Article title
Cleaned article body

Next article
Its cleaned article text

```

The normal full-language build targets up to **16** top-level shards (or fewer when a Wikimedia dump exposes fewer distinct range files), converting at most two root shards concurrently. A failing shard is split into two independent groups and retried recursively up to three levels; suffixes `a` and `b` indicate successive splits:

- `enwiki_03_of_16.txt.gz`: root shard 3 completed intact.
- `enwiki_03a_of_16.txt.gz` and `enwiki_03b_of_16.txt.gz`: the two halves of failed root shard 3.
- `enwiki_03ba_of_16.txt.gz` and `enwiki_03bb_of_16.txt.gz`: a further split of the `03b` half.

These are **disjoint subsets**, not multiple copies of the same archive. Each dataset build includes an explicit coverage manifest; if recovery is incomplete, the dataset card identifies it as **PARTIAL**.

## Run locally

```bash
python -m pip install -r REQUIREMENTS.TXT
python wiki_to_text.py dump.xml.bz2 output --shards 1 --prefix wiki
```

For automation, use [Build all Wikipedia languages](.github/workflows/rebuild-languages.yml) or [Rebuild Wikipedia shards](.github/workflows/rebuild-full.yml). The source range downloader is [`scripts/download_page_ranges.py`](scripts/download_page_ranges.py), and adaptive recovery is [`scripts/convert_adaptive.py`](scripts/convert_adaptive.py).

The pipelines upload successfully completed archives even when other shards fail, and attempt publication to Hugging Face and Kaggle separately. Temporary GitHub Actions shard artifacts are automatically deleted after at least one publishing destination confirms success. If every publisher fails, the artifacts are retained briefly to allow recovery.

## Attribution and licensing

- **Source content:** [Wikimedia dumps](https://dumps.wikimedia.org/), subject to Wikipedia's applicable Creative Commons Attribution-ShareAlike license and attribution requirements.
- **This maintained fork and dataset build pipeline:** [cheeseonamonkey/PlainTextWikipedia](https://github.com/cheeseonamonkey/PlainTextWikipedia).
- **Original project:** [daveshap/PlainTextWikipedia](https://github.com/daveshap/PlainTextWikipedia).

The converter's MIT license does not replace the license governing Wikipedia article content.
