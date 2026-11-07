"""Shared Unicode tokenization and dependency-backed multilingual word reduction."""
from __future__ import annotations

import unicodedata
from collections import Counter

import regex
import Stemmer
from stopwordsiso import stopwords

FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                      "‐": "-", "‑": "-", "‒": "-", "–": "-",
                      "—": "-", "―": "-", "\u00a0": " ", "…": "..."})
WORD = regex.compile(r"[\p{L}\p{N}\p{M}]+(?:['-][\p{L}\p{N}\p{M}]+)*")
MAX_DF_RATIO = .5


def fold(text):
    return unicodedata.normalize('NFKC', text.translate(FOLD)).casefold()


def tokens(text):
    result = []
    for word in WORD.findall(fold(text)):
        if len(word) > 1 and any('\u3040' <= ch <= '\u30ff' or '\u3400' <= ch <= '\u9fff'
                                 or '\u0e00' <= ch <= '\u0e7f' for ch in word):
            result.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            result.append(word)
    return result


class LexicalNormalizer:
    def __init__(self, code):
        iso = code.split('_', 1)[0].split('-', 1)[0]
        self.iso = {'simple': 'en', 'nb': 'no', 'nn': 'no'}.get(iso, iso)
        try:
            self.stemmer = Stemmer.Stemmer(self.iso)
            self.stemmer.maxCacheSize = 65536
        except KeyError:
            self.stemmer = None
        self.stopwords = frozenset(fold(word) for word in (stopwords(self.iso) or ()))
        self.stop_stems = frozenset(self.stem(word) for word in self.stopwords)

    def stem(self, word):
        return self.stemmer.stemWord(word) if self.stemmer else word

    def words(self, words, vocabulary=None):
        for word in words:
            if word in self.stopwords or not any(ch.isalpha() for ch in word):
                continue
            stem = self.stem(word)
            if (stem and stem not in self.stop_stems and len(stem) <= 64
                    and (vocabulary is None or stem in vocabulary)):
                yield stem

    def counts(self, counts):
        result = Counter()
        for word, count in counts.items():
            for stem in self.words([word]):
                result[stem] += count
        return result

    def config(self):
        return dict(language=self.iso, normalization='NFKC+casefold+quote_dash_fold',
                    morphology='Snowball/PyStemmer' if self.stemmer else 'unavailable; normalized surface forms',
                    stemmer_version='3.0.0', stopwords='stopwordsiso==0.7.1',
                    stopword_count=len(self.stopwords), max_df_ratio=MAX_DF_RATIO,
                    unsupported_languages='retain normalized tokens; no article is omitted')
