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


def test_general_knowledge_answer_passes_through_unchanged():
    answer, overridden = validate_answer(
        "What is the capital of France?", "Paris.", FACTS
    )
    assert answer == "Paris."
    assert overridden is False


def test_fabricated_ecobrew_answer_still_overridden_despite_keyword_gate():
    answer, overridden = validate_answer(
        "Does the EcoBrew Mini support Bluetooth?", "Yes, the EcoBrew Mini supports Bluetooth 5.0.", FACTS
    )
    assert answer == ABSTAIN
    assert overridden is True


def test_fabricated_price_matching_unrelated_facts_accept_string_is_still_overridden():
    # "$149.99" contains "149", which is the real EcoBrew Pro's accept-string --
    # but the question asks about a fictional "XL" variant, so it must not pass.
    answer, overridden = validate_answer(
        "What does the EcoBrew XL cost?", "The EcoBrew XL costs $149.99.", FACTS
    )
    assert answer == ABSTAIN
    assert overridden is True


def test_known_variant_with_correct_answer_still_passes_through():
    answer, overridden = validate_answer(
        "What does the EcoBrew Max cost?", "The EcoBrew Max costs $219.", FACTS
    )
    assert answer == "The EcoBrew Max costs $219."
    assert overridden is False
