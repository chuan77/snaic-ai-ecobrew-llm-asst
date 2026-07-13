from data.facts import FACTS
from data.generate import ABSTAIN
from evaluation.harness import _is_abstain, _norm

ECOBREW_KEYWORDS = ("ecobrew", "verdant", "greencup", "sprout", "homebase")


def validate_answer(question, raw_answer, facts=FACTS):
    if _is_abstain(raw_answer):
        return raw_answer, False

    question_lower = question.lower()
    if not any(keyword in question_lower for keyword in ECOBREW_KEYWORDS):
        return raw_answer, False  # not an EcoBrew-domain question; nothing to fact-check

    normalized = _norm(raw_answer)
    for fact in facts:
        if any(_norm(accept) in normalized for accept in fact["accept"]):
            return raw_answer, False

    return ABSTAIN, True
