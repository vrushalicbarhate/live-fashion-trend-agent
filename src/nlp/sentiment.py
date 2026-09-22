"""
Sentiment Analysis (synopsis §5.5, §9.3).

Runs local sentiment inference on the exact sentence each mention came
from (not the whole post — see module docstring reasoning in the repo
notes). Keeps all three class probabilities rather than just the top
label, because the Trend Score formula needs a continuous polarity
value: polarity = positive_prob - negative_prob, in [-1, 1].

Default model: cardiffnlp/twitter-roberta-base-sentiment-latest
(social-media-tuned, CC BY 4.0) — see settings.yaml sentiment_model.name.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from src.nlp.entities import Mention


class SentimentClassifier(Protocol):
    """
    Minimal interface we depend on from a Hugging Face
    `pipeline("text-classification", top_k=None)` call, so tests can
    inject a fake classifier instead of loading real model weights.

    Expected shape: given a list of N texts, returns a list of N lists,
    each containing one {"label": ..., "score": ...} dict per class.
    """

    def __call__(self, texts: list[str]) -> list[list[dict]]: ...


def _extract_probabilities(class_scores: list[dict]) -> dict[str, float]:
    """
    Normalizes a model's per-class output into {positive, neutral, negative}
    regardless of label casing (cardiffnlp uses lowercase; some models
    use LABEL_0/1/2 or Title Case).
    """
    probs = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for entry in class_scores:
        label = entry["label"].lower()
        if label in probs:
            probs[label] = entry["score"]
    return probs


def score_mentions(
    mentions: list[Mention],
    classifier: SentimentClassifier,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Runs sentiment inference over each mention's sentence in batches.
    Returns a DataFrame with one row per mention: item_id, sentence,
    positive, neutral, negative, polarity.
    """
    if not mentions:
        return pd.DataFrame(
            columns=["item_id", "sentence", "positive", "neutral", "negative", "polarity"]
        )

    sentences = [m.sentence for m in mentions]
    rows = []

    for start in range(0, len(sentences), batch_size):
        batch_texts = sentences[start : start + batch_size]
        batch_mentions = mentions[start : start + batch_size]
        batch_results = classifier(batch_texts)

        for mention, class_scores in zip(batch_mentions, batch_results):
            probs = _extract_probabilities(class_scores)
            polarity = probs["positive"] - probs["negative"]
            rows.append(
                {
                    "item_id": mention.item_id,
                    "sentence": mention.sentence,
                    "positive": probs["positive"],
                    "neutral": probs["neutral"],
                    "negative": probs["negative"],
                    "polarity": polarity,
                }
            )

    return pd.DataFrame(rows)
