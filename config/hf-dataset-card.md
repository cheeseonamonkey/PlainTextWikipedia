---
language:
- __LANG_CODE__
license: cc-by-sa-3.0
pretty_name: Plaintext Wikipedia (__LANG_NAME__)
---

# Plaintext Wikipedia (__LANG_NAME__)

Compressed plaintext articles extracted from [__LANG_NAME__ Wikipedia](__SOURCE_URL__) dumps. Each article title precedes its cleaned text, and every shard is independently compressed.

## Dataset Structure

Each file is a gzipped text shard. Inside, articles are formatted as:

```text
Article Title
Article body text on subsequent lines.
```

Shard files are named `{__LANG_CODE__}_{id:02d}_of_{count:02d}.txt.gz`.

## Usage

You can load the full dataset using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

ds = load_dataset("ffatty/plaintext-wikipedia-full-__LANG_CODE__")
for article in ds["train"]:
    print(article["title"], article["body"][:100])
```

Or read raw shards directly:

```python
import gzip

with gzip.open("enwiki_01_of_10.txt.gz", "rt", encoding="utf-8") as f:
    for line in f:
        print(line)
```

## Citation / Attribution

If you use this dataset, credit:

1.  [cheeseonamonkey/PlainTextWikipedia](__REPO_URL__) — maintained fork that generated this release.
2.  [daveshap/PlainTextWikipedia](__UPSTREAM_URL__) — original upstream project (unmaintained).
3.  [Wikimedia Foundation](https://www.wikimedia.org/) — source data provider.

## License

Wikipedia text is available under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The converter code is MIT licensed.

---
Built by [cheeseonamonkey/PlainTextWikipedia](__REPO_URL__). Forked from [daveshap/PlainTextWikipedia](__UPSTREAM_URL__) *(not maintained)*. Source data: official Wikimedia dumps.
