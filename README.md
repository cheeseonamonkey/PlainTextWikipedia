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

## Additional language imports (launch batch)

The following **42 additional editions** are configured for the first import. These are target URLs, **not proof that a dataset has been published**. Some may remain pending, fail preflight because an October dump is unavailable, or require publisher retries.

| Wikipedia edition | Hugging Face destination | Kaggle destination |
|---|---|---|
| Arabic (`arwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-arabic) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-arabic) |
| Japanese (`jawiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-japanese) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-japanese) |
| Serbo-Croatian (`shwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-serbo-croatian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-serbo-croatian) |
| Persian (`fawiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-persian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-persian) |
| Swedish (`svwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-swedish) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-swedish) |
| Ukrainian (`ukwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-ukrainian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-ukrainian) |
| Swahili (`swwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-swahili) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-swahili) |
| Korean (`kowiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-korean) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-korean) |
| Esperanto (`eowiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-esperanto) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-esperanto) |
| Scots (`scowiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-scots) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-scots) |
| Toki Pona (`tokwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-toki-pona) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-toki-pona) |
| Hindi (`hiwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-hindi) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-hindi) |
| Bengali (`bnwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-bengali) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-bengali) |
| Urdu (`urwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-urdu) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-urdu) |
| Navajo (`nvwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-navajo) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-navajo) |
| Cherokee (`chrwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-cherokee) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-cherokee) |
| Basque (`euwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-basque) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-basque) |
| Latin (`lawiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-latin) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-latin) |
| Telugu (`tewiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-telugu) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-telugu) |
| Tok Pisin (`tpiwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-tok-pisin) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-tok-pisin) |
| Nigerian Pidgin (`pcmwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-nigerian-pidgin) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-nigerian-pidgin) |
| Vietnamese (`viwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-vietnamese) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-vietnamese) |
| Marathi (`mrwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-marathi) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-marathi) |
| Hausa (`hawiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-hausa) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-hausa) |
| Ido (`iowiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-ido) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-ido) |
| Volapük (`vowiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-volapuk) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-volapuk) |
| Lojban (`jbowiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-lojban) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-lojban) |
| Gothic (`gotwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-gothic) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-gothic) |
| Old English (`angwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-old-english) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-old-english) |
| Classical Chinese (`zh_classicalwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-classical-chinese) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-classical-chinese) |
| Aramaic (Syriac) (`arcwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-aramaic-syriac) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-aramaic-syriac) |
| Dutch (`nlwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-dutch) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-dutch) |
| Polish (`plwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-polish) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-polish) |
| Waray (`warwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-waray) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-waray) |
| Chechen (`cewiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-chechen) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-chechen) |
| Serbian (`srwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-serbian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-serbian) |
| Turkish (`trwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-turkish) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-turkish) |
| Greek (`elwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-greek) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-greek) |
| Czech (`cswiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-czech) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-czech) |
| Welsh (`cywiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-welsh) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-welsh) |
| Thai (`thwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-thai) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-thai) |
| Cebuano (`cebwiki`) | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-cebuano) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-cebuano) |

The import batch is defined in [`bootstrap-additional-languages.yml`](.github/workflows/bootstrap-additional-languages.yml) and [`additional-languages.json`](config/additional-languages.json). It reuses the existing maximum of 16 initial shards, a/b recursive recovery, and post-publication artifact cleanup. Cebuano is included; its millions of mostly bot-created articles may be repetitive compared with human-written corpora.

**Not included as active Wikimedia Wikipedia dumps:** Klingon (closed), Cree (closed), ǃXóõ / Taa (no identified active edition), and Tigre (no verified active edition). Scots uses the `scowiki` edition, and “Pidgin” includes both Tok Pisin and Nigerian Pidgin. Tigrinya is a different language from Tigre.

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
