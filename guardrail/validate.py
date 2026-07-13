import re

from data.facts import FACTS
from data.generate import ABSTAIN
from evaluation.harness import _is_abstain, _norm

ECOBREW_KEYWORDS = ("ecobrew", "verdant", "greencup", "sprout")

_WORD_RE = re.compile(r"[a-z0-9]+")
# Only capitalized tokens right after "EcoBrew" are treated as candidate product
# names (e.g. "EcoBrew Pro", "EcoBrew XL"). Lowercase continuations are almost
# always ordinary sentence words ("ecobrew need descaling"), not model names.
_VARIANT_RE = re.compile(r"EcoBrew\+?\s+([A-Z][a-zA-Z]*)")


def _corpus_words(facts):
    words = set()
    for fact in facts:
        words |= set(_WORD_RE.findall(fact["question"].lower()))
        words |= set(_WORD_RE.findall(fact["answer"].lower()))
    return words


def _mentioned_variants(text):
    return {word.lower() for word in _VARIANT_RE.findall(text)}


def _mentions_unknown_variant(question, facts):
    corpus_words = _corpus_words(facts)
    return any(variant not in corpus_words for variant in _mentioned_variants(question))


def _relevant_facts(question, facts):
    """Narrow the accept-list check to facts about the specific variant asked
    about, so e.g. a fabricated Max price can't pass by coincidentally matching
    the Pro's accept-string. Falls back to the full table when the question
    doesn't name a specific variant, or names one no fact happens to mention
    under this simple extraction (safe default: same as the old behavior)."""
    variants = _mentioned_variants(question)
    if not variants:
        return facts
    scoped = [
        fact for fact in facts
        if variants & (_mentioned_variants(fact["question"]) | _mentioned_variants(fact["answer"]))
    ]
    return scoped or facts


def validate_answer(question, raw_answer, facts=FACTS):
    if _is_abstain(raw_answer):
        return raw_answer, False

    question_lower = question.lower()
    if not any(keyword in question_lower for keyword in ECOBREW_KEYWORDS):
        return raw_answer, False  # not an EcoBrew-domain question; nothing to fact-check

    if _mentions_unknown_variant(question, facts):
        # e.g. "the EcoBrew XL" -- a product name never seen in any fact, so no
        # accept-list check below can be trusted even if it happens to match.
        return ABSTAIN, True

    normalized = _norm(raw_answer)
    for fact in _relevant_facts(question, facts):
        if any(_norm(accept) in normalized for accept in fact["accept"]):
            return raw_answer, False

    return ABSTAIN, True
