from evaluation.harness import _is_abstain, _norm, evaluate

QUESTIONS = [
    {"type": "recall", "question": "What color is the sky?", "accept": ["blue"]},
    {"type": "unanswerable", "question": "What is the meaning of life?", "accept": []},
    {"type": "general", "question": "What is 2 + 2?", "accept": ["4"]},
]


def test_norm_lowercases_and_strips_commas():
    assert _norm("Blue, obviously") == "blue obviously"


def test_is_abstain_detects_exact_phrase():
    assert _is_abstain("I don't have that information.")


def test_is_abstain_rejects_confident_answer():
    assert not _is_abstain("The sky is blue.")


def test_evaluate_perfect_predictor():
    def predict(question):
        return {
            "What color is the sky?": "The sky is blue.",
            "What is the meaning of life?": "I don't have that information.",
            "What is 2 + 2?": "4",
        }[question]

    scores = evaluate(predict, QUESTIONS)
    assert scores == {"recall": 1.0, "abstain": 1.0, "general": 1.0}


def test_evaluate_hallucinating_predictor_scores_zero_abstain():
    def predict(question):
        return "42, obviously." if question == "What is the meaning of life?" else "The sky is blue."

    scores = evaluate(predict, QUESTIONS)
    assert scores["abstain"] == 0.0
