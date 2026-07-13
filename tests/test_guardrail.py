from data.facts import FACTS
from data.generate import ABSTAIN
from guardrail.validate import validate_answer


def test_known_fact_answer_passes_through_unchanged():
    answer, overridden = validate_answer(
        "What does the EcoBrew One cost?", "The EcoBrew One costs $89.", FACTS
    )
    assert answer == "The EcoBrew One costs $89."
    assert overridden is False


def test_abstain_answer_passes_through_unchanged():
    answer, overridden = validate_answer(
        "How much does the EcoBrew Mini cost?", ABSTAIN, FACTS
    )
    assert answer == ABSTAIN
    assert overridden is False


def test_fabricated_answer_gets_overridden_to_abstain():
    answer, overridden = validate_answer(
        "How much does the EcoBrew Mini cost?", "The EcoBrew Mini costs $59.", FACTS
    )
    assert answer == ABSTAIN
    assert overridden is True
