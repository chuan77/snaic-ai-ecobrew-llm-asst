def _norm(text):
    return text.lower().replace(",", "")


def _is_abstain(answer):
    normalized = answer.lower()
    phrases = (
        "don't have that information",
        "do not have that information",
        "don't have that",
        "not have that information",
    )
    return any(phrase in normalized for phrase in phrases)


def evaluate(predict_fn, questions):
    recall_qs = [q for q in questions if q["type"] == "recall"]
    unanswerable_qs = [q for q in questions if q["type"] == "unanswerable"]
    general_qs = [q for q in questions if q["type"] == "general"]

    def hit_rate(qs):
        hits = 0
        for q in qs:
            answer = _norm(predict_fn(q["question"]))
            if any(_norm(accept) in answer for accept in q["accept"]):
                hits += 1
        return hits / len(qs)

    recall = hit_rate(recall_qs)
    general = hit_rate(general_qs)
    abstain = sum(_is_abstain(predict_fn(q["question"])) for q in unanswerable_qs) / len(unanswerable_qs)

    return {"recall": recall, "abstain": abstain, "general": general}
