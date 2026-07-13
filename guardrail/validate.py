from data.facts import FACTS
from data.generate import ABSTAIN
from evaluation.harness import _is_abstain, _norm


def validate_answer(question, raw_answer, facts=FACTS):
    if _is_abstain(raw_answer):
        return raw_answer, False

    normalized = _norm(raw_answer)
    for fact in facts:
        if any(_norm(accept) in normalized for accept in fact["accept"]):
            return raw_answer, False

    return ABSTAIN, True
