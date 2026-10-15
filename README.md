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

**Not included as active Wikipedia editions:** Klingon and Cree (closed), and Taa/ǃXóõ (no identified active edition). **Tigre (`tigwiki`) is active** and included in the new batch; Tigrinya (`tiwiki`) is a separate language. Scots uses `scowiki`, and “Pidgin” includes both Tok Pisin and Nigerian Pidgin.

## Hebrew and language-diversity import batch

An additional **73 Wikipedia editions** have been configured for an initial import, emphasizing Indigenous, historically marginalized, minority, or revitalized languages. **These are proposed publishing destinations, not proof an upload has completed.** Dataset cards and `build-manifest.json` show any missing coverage.

The grouping below is for discovery and **does not represent an authoritative endangered-language classification**. Hebrew is included because it was specifically requested, and is **not considered an endangered modern language**. A small encyclopedia is not necessarily a dying language; some of these languages have substantial active speaker communities.

| Wikipedia edition | Discovery group | Hugging Face destination | Kaggle destination |
|---|---|---|---|
| Hebrew (`hewiki`) | Additional requested language | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-hebrew) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-hebrew) |
| Atikamekw (`atjwiki`) | Indigenous Americas | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-atikamekw) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-atikamekw) |
| Cheyenne (`chywiki`) | Indigenous Americas | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-cheyenne) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-cheyenne) |
| Inuktitut (`iuwiki`) | Indigenous Americas | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-inuktitut) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-inuktitut) |
| Inupiaq (`ikwiki`) | Indigenous Americas | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-inupiaq) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-inupiaq) |
| Nahuatl (`nahwiki`) | Indigenous Americas | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-nahuatl) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-nahuatl) |
| Pipil (Nawat) (`pplwiki`) | Indigenous Americas | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-pipil-nawat) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-pipil-nawat) |
| Quechua (`quwiki`) | Indigenous Americas | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-quechua) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-quechua) |
| Aymara (`aywiki`) | Indigenous Americas | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-aymara) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-aymara) |
| Guarani (`gnwiki`) | Indigenous Americas | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-guarani) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-guarani) |
| Hawaiian (`hawwiki`) | Pacific and Polynesian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-hawaiian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-hawaiian) |
| Maori (`miwiki`) | Pacific and Polynesian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-maori) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-maori) |
| Tahitian (`tywiki`) | Pacific and Polynesian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-tahitian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-tahitian) |
| Tongan (`towiki`) | Pacific and Polynesian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-tongan) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-tongan) |
| Samoan (`smwiki`) | Pacific and Polynesian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-samoan) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-samoan) |
| Central Dusun (`dtpwiki`) | Pacific and Polynesian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-central-dusun) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-central-dusun) |
| Sakizaya (`szywiki`) | Indigenous Taiwan | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-sakizaya) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-sakizaya) |
| Atayal (`taywiki`) | Indigenous Taiwan | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-atayal) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-atayal) |
| Seediq (`trvwiki`) | Indigenous Taiwan | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-seediq) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-seediq) |
| Amis (`amiwiki`) | Indigenous Taiwan | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-amis) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-amis) |
| Paiwan (`pwnwiki`) | Indigenous Taiwan | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-paiwan) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-paiwan) |
| Breton (`brwiki`) | Celtic and revived | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-breton) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-breton) |
| Irish (`gawiki`) | Celtic and revived | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-irish) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-irish) |
| Scottish Gaelic (`gdwiki`) | Celtic and revived | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-scottish-gaelic) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-scottish-gaelic) |
| Cornish (`kwwiki`) | Celtic and revived | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-cornish) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-cornish) |
| Manx (`gvwiki`) | Celtic and revived | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-manx) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-manx) |
| Northern Sami (`sewiki`) | Uralic and Sámi | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-northern-sami) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-northern-sami) |
| Inari Sami (`smnwiki`) | Uralic and Sámi | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-inari-sami) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-inari-sami) |
| Veps (`vepwiki`) | Uralic and Sámi | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-veps) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-veps) |
| Udmurt (`udmwiki`) | Uralic and Sámi | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-udmurt) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-udmurt) |
| Komi (`kvwiki`) | Uralic and Sámi | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-komi) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-komi) |
| Livvi-Karelian (`olowiki`) | Uralic and Sámi | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-livvi-karelian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-livvi-karelian) |
| Komi-Permyak (`koiwiki`) | Uralic and Sámi | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-komi-permyak) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-komi-permyak) |
| Meadow Mari (`mhrwiki`) | Uralic and Sámi | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-meadow-mari) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-meadow-mari) |
| Hill Mari (`mrjwiki`) | Uralic and Sámi | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-hill-mari) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-hill-mari) |
| Erzya (`myvwiki`) | Uralic and Sámi | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-erzya) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-erzya) |
| Moksha (`mdfwiki`) | Uralic and Sámi | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-moksha) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-moksha) |
| Buryat (`bxrwiki`) | Northern Asian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-buryat) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-buryat) |
| Kalmyk (`xalwiki`) | Northern Asian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-kalmyk) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-kalmyk) |
| Sakha (Yakut) (`sahwiki`) | Northern Asian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-sakha) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-sakha) |
| Tuvan (`tyvwiki`) | Northern Asian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-tuvan) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-tuvan) |
| Altai (`altwiki`) | Northern Asian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-altai) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-altai) |
| Gagauz (`gagwiki`) | Northern Asian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-gagauz) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-gagauz) |
| Karachay-Balkar (`krcwiki`) | Northern Asian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-karachay-balkar) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-karachay-balkar) |
| Occitan (`ocwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-occitan) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-occitan) |
| Aragonese (`anwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-aragonese) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-aragonese) |
| Asturian (`astwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-asturian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-asturian) |
| Lombard (`lmowiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-lombard) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-lombard) |
| Piedmontese (`pmswiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-piedmontese) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-piedmontese) |
| Venetian (`vecwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-venetian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-venetian) |
| Neapolitan (`napwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-neapolitan) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-neapolitan) |
| Emilian-Romagnol (`emlwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-emilian-romagnol) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-emilian-romagnol) |
| Ligurian (`lijwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-ligurian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-ligurian) |
| Sardinian (`scwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-sardinian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-sardinian) |
| Picard (`pcdwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-picard) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-picard) |
| Friulian (`furwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-friulian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-friulian) |
| Arpitan (`frpwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-arpitan) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-arpitan) |
| Norman (`nrmwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-norman) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-norman) |
| Mirandese (`mwlwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-mirandese) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-mirandese) |
| Extremaduran (`extwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-extremaduran) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-extremaduran) |
| Ladino (`ladwiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-ladino) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-ladino) |
| Corsican (`cowiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-corsican) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-corsican) |
| Walloon (`wawiki`) | European regional and heritage | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-walloon) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-walloon) |
| Standard Moroccan Amazigh (`zghwiki`) | African and Caucasian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-standard-moroccan-amazigh) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-standard-moroccan-amazigh) |
| Shilha (`shiwiki`) | African and Caucasian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-shilha) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-shilha) |
| Kabyle (`kabwiki`) | African and Caucasian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-kabyle) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-kabyle) |
| Abkhaz (`abwiki`) | African and Caucasian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-abkhaz) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-abkhaz) |
| Kabardian (`kbdwiki`) | African and Caucasian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-kabardian) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-kabardian) |
| Adyghe (`adywiki`) | African and Caucasian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-adyghe) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-adyghe) |
| Tigrinya (`tiwiki`) | African and Caucasian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-tigrinya) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-tigrinya) |
| Tigre (`tigwiki`) | African and Caucasian | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-tigre) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-tigre) |
| Romani (`rmywiki`) | Other vulnerable or underrepresented | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-romani) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-romani) |
| Uyghur (`ugwiki`) | Other vulnerable or underrepresented | [Hugging Face](https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-uyghur) | [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-uyghur) |

See [the one-shot 73-edition bootstrap workflow](.github/workflows/bootstrap-language-diversity.yml). It reuses the maximum of 16 initial shards, binary recovery and automatic artifact deletion after confirmed publication. Existing datasets are not rebuilt by this additional-language workflow.

For linguistic research, distinguish human-authored material, translations, templated/bot-generated pages and stubs. Public Wikipedia text is useful but is **not** a replacement for community-produced or community-authorized language documentation.

## Global coverage expansion — 58 more editions (October 2026)

The [58-language bootstrap workflow](.github/workflows/bootstrap-global-oct15.yml) adds new active Wikipedia editions verified against Wikimedia's site matrix. This is **independent** of the earlier 36-language October batch; no previously configured edition is repeated.

| Coverage group | New language editions |
|---|---|
| South/Central Asia | Kyrgyz, Tajik, Sinhala, Gujarati, Assamese, Sindhi, Pashto, Kurdish, Tatar, Bashkir |
| Southeast Asia | Burmese, Khmer, Lao, Javanese, Sundanese, Minangkabau, Acehnese, Balinese, Central Bikol, Ilocano, Kapampangan, Pangasinan |
| Africa | Xhosa, Shona, Southern Sotho, Twi, Oromo, Tsonga, Tswana, Kinyarwanda, Luganda, Northern Sotho, Swati, Venda, Chichewa, Lingala, Kongo, Kirundi, Malagasy, Wolof, Fulah, Dagbani, Dinka, Bambara, Ewe, Fon |
| Americas/Pacific | Haitian Creole, Papiamento, Jamaican Patois, Fijian, Bislama, Chamorro |
| Europe | Catalan, Galician, West Frisian, Luxembourgish, Maltese, Macedonian |

All 58 official project codes, exact dataset slugs, and display names live in [config/additional-languages.json](config/additional-languages.json). Their configured destinations follow `https://huggingface.co/datasets/ffatty/plaintext-wikipedia-full-<slug>` and `https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-<slug>`.

**Coverage vs publication:** An active Wikimedia edition does not guarantee a completed dump or successfully published corpus. Each CI job checks available archives, preserves missing ranges and failure information, and publishes available validated data independently. Newly split output files use the **20 MiB minimum compressed size** rule (a small entire dataset may remain smaller than 20 MiB). The workflow runs at most two language editions at a time and does not cancel the earlier batch.

## Source discovery and output sharding

The number of **original Wikimedia XML source files** is not the number of
**published text shards**. Small Wikipedias may legitimately publish a single
XML source file. Our publisher now targets **16 independently compressed text
archives**, when there are enough articles, without re-downloading that XML
sixteen times. Each repacked archive uses a `_part01_of_16` suffix and is
verified against the complete original article sequence before the original is
removed. The build manifest records how each output file was partitioned.

When legacy `/latest/` XML isn't available, the planner probes Wikimedia's
new [MediaWiki Content File Exports](https://wikitech.wikimedia.org/wiki/MediaWiki_Content_File_Exports).
These monthly `mediawiki_content_current` snapshots contain current revision
XML in independent page-ID ranges. We select a completed monthly snapshot only
when its `SHA256SUMS` is present, and **verify every downloaded file against
that SHA-256**. Any failed or corrupt download is reflected in the partial
coverage manifest; it is never silently treated as complete.

Do **not** remove a language merely because the official source is one file
or because a current-month dump hasn't finished. Previous complete monthly
snapshots are valid alternatives, marked with their snapshot date. We remove
only malformed generated shard files during validation; deleting existing
published datasets requires separate explicit review.

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

**Compressed output sizing:** New publication runs split completed archives only when they can produce shards of at least **20 MiB compressed**, and never produce more than **16** files through the post-conversion repacker. A 3 MiB or 35 MiB small-language edition generally stays a single archive; a much larger archive can be split while preserving exact article bytes. Historic releases repacked before this rule may still contain tiny gzip files.

**Kaggle backfill:** Authenticated machines can publish already-available, SHA-256-verified Hugging Face files directly using `python scripts/kaggle_backfill.py swahili hindi korean --output /root/wiki-kaggle-stage`, avoiding another Wikimedia dump conversion. CI checks the actual Kaggle credentials first and supports both legacy API keys and modern Kaggle access tokens.

**Source files are not output files.** Even when Wikimedia supplies just one source XML archive, the publisher can split its already converted articles into up to **16 gzip files** without redownloading or reconverting the XML. Those post-conversion archives receive a `_partNN_of_MM` suffix. SHA-256 reconstruction checks that no article bytes were lost or duplicated, and very small corpora are not padded with empty archives. Valid output from incomplete roots may also be split; their manifest retains `complete: false`.

This applies to **new publishing runs**. Older Hugging Face or Kaggle releases that were uploaded before the repacking workflow retain their old archive layout until they are republished.

**Hugging Face legacy migration:** [Repack existing Hugging Face Wikipedia archives](.github/workflows/repack-existing-hf.yml) downloads the published gzip files once, SHA-256-verifies an article-exact split, uploads replacements before removing old archives, and retains original completeness/missing-range claims. It also supports manually selecting one existing `ffatty` dataset. Datasets with very large archives or unverified size metadata are skipped rather than risking a destructive upload. Kaggle archives are not changed by this separate migration.

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
