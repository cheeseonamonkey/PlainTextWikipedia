"""Bounded, explicitly scoped article sentiment, readability and lexical diagnostics.

Sentiment and US grade-level readability are English-only. Lexical continuity
is not natural-language inference and must not be presented as contradiction.
"""
from __future__ import annotations

import math


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class Diagnostics:
    def __init__(self, language: str):
        self.english = language in {"en", "simple"}
        self.sentiment = None
        self.readability = None
        if self.english:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            import textstat
            self.sentiment = SentimentIntensityAnalyzer()
            self.readability = textstat

    def calculate(self, body: str, *, rare_ratio: float, digit_ratio: float,
                  mean_token_chars: float, mean_idf: float, tokenizer, splitter) -> dict:
        sample = body if len(body) <= 4096 else body[:3072] + " " + body[-1024:]
        polarity = self.sentiment.polarity_scores(sample) if self.sentiment and sample else None
        # Flesch-Kincaid formulas and VADER lexicon are not language independent.
        reading_sample = body[:2048] if self.readability and body else ""
        grade = (float(self.readability.flesch_kincaid_grade(reading_sample))
                 if reading_sample else None)
        ease = (float(self.readability.flesch_reading_ease(reading_sample))
                if reading_sample else None)
        # Topic continuity over adjacent sentences, NOT contradiction detection.
        sentence_sets = [set(tokenizer(s)) for s in splitter.split(body[:12_000])[:48] if s.strip()]
        connections = [len(a & b) / max(1, len(a | b))
                       for a, b in zip(sentence_sets, sentence_sets[1:]) if a and b]
        cohesion = sum(connections) / len(connections) if connections else None
        technical = (0.4 * clamp(rare_ratio / .25) +
                     0.25 * clamp((mean_token_chars - 4) / 6) +
                     0.2 * clamp((mean_idf - 1) / 7) +
                     0.15 * clamp(digit_ratio / .10))
        compound = polarity['compound'] if polarity else None
        return {
            "vader_compound": compound,
            "vader_positive": polarity['pos'] if polarity else None,
            "vader_neutral": polarity['neu'] if polarity else None,
            "vader_negative": polarity['neg'] if polarity else None,
            "sentiment_valence_01": (compound + 1) / 2 if compound is not None else None,
            "sentiment_intensity_01": abs(compound) if compound is not None else None,
            "sentiment_sample_chars": len(sample) if polarity else 0,
            "reading_grade": grade,
            "reading_ease": ease,
            "reading_difficulty_01": clamp(grade / 20) if grade is not None else None,
            "reading_sample_chars": len(reading_sample),
            "technicality_proxy_01": technical,
            "lexical_cohesion_01": cohesion,
            "contradiction_score": None,  # Requires an NLI model; not inferred from lexical similarity.
        }
