from src.nlp.entities import Mention
from src.nlp.sentiment import score_mentions


class FakeClassifier:
    """
    Stands in for a HF pipeline. `canned` maps sentence text -> the
    per-class score list that pipeline would return for it.
    """

    def __init__(self, canned: dict[str, list[dict]]):
        self.canned = canned
        self.call_sizes: list[int] = []  # records how many texts came in per call

    def __call__(self, texts: list[str]) -> list[list[dict]]:
        self.call_sizes.append(len(texts))
        return [self.canned[t] for t in texts]


def make_mention(item_id, sentence):
    return Mention(item_id=item_id, sentence=sentence, product="p", material="m", color="c")


def test_polarity_is_positive_minus_negative():
    mentions = [make_mention("C001", "I love this jacket")]
    classifier = FakeClassifier(
        {
            "I love this jacket": [
                {"label": "positive", "score": 0.8},
                {"label": "neutral", "score": 0.15},
                {"label": "negative", "score": 0.05},
            ]
        }
    )
    df = score_mentions(mentions, classifier)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["positive"] == 0.8
    assert row["negative"] == 0.05
    assert abs(row["polarity"] - 0.75) < 1e-9


def test_handles_uppercase_and_label_variants():
    mentions = [make_mention("C001", "This is fine I guess")]
    classifier = FakeClassifier(
        {
            "This is fine I guess": [
                {"label": "Positive", "score": 0.2},
                {"label": "Neutral", "score": 0.6},
                {"label": "Negative", "score": 0.2},
            ]
        }
    )
    df = score_mentions(mentions, classifier)
    row = df.iloc[0]
    assert row["neutral"] == 0.6
    assert row["polarity"] == 0.0  # 0.2 - 0.2


def test_batches_respect_batch_size():
    mentions = [make_mention("C001", f"sentence {i}") for i in range(5)]
    canned = {
        f"sentence {i}": [
            {"label": "positive", "score": 0.5},
            {"label": "neutral", "score": 0.3},
            {"label": "negative", "score": 0.2},
        ]
        for i in range(5)
    }
    classifier = FakeClassifier(canned)

    df = score_mentions(mentions, classifier, batch_size=2)

    assert len(df) == 5
    assert classifier.call_sizes == [2, 2, 1]  # 5 items in batches of 2


def test_empty_mentions_returns_empty_dataframe_with_expected_columns():
    classifier = FakeClassifier({})
    df = score_mentions([], classifier)

    assert df.empty
    assert list(df.columns) == ["item_id", "sentence", "positive", "neutral", "negative", "polarity"]
