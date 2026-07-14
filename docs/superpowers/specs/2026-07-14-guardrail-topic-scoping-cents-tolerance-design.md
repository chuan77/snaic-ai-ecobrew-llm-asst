# Design: Guardrail Topic-Scoping + Gated Cents-Tolerance

**Date:** 2026-07-14
**Status:** Approved for planning

## 1. Context

Manual re-verification of the previous guardrail fix (domain-gate widening + numeric boundary matching,
`docs/superpowers/specs/2026-07-14-guardrail-domain-gate-numeric-match-fix-design.md`) confirmed the
original reported bug is fixed, but surfaced two further findings while spot-checking real questions:

1. **Cross-category coincidental match (new bug).** "What is EcoBrew's return policy window?" (real answer:
   45-day window) got the model's confused answer "EcoBrew allows returns within two years of purchase." —
   which fabricates the *warranty* fact's content ("2-year") instead of the return policy. This slipped
   through the guardrail uncaught: the question names no product variant, so `_relevant_facts` falls back
   to the *entire* fact table, and the fabricated text's "two years" contains the warranty fact's
   non-numeric accept-string "two year" as a plain substring — a coincidental match, just like the original
   bug, but on the non-numeric side and across unrelated fact *categories* rather than product variants.

2. **A side-effect of the previous fix's strictness.** "What does the EcoBrew Pro cost?" (real: $149) got
   the model's answer "$149.99" — the right number with spurious cents appended. The prior fix's boundary-aware
   numeric matching now correctly treats `149.99 != 149` and abstains, which is defensible in isolation
   but means a legitimate, previously-passing recall question now abstains due to the model's own minor
   imprecision, not an actual hallucination.

These two findings share a root cause and a resolution, discussed with the user directly: **cents-tolerance
is only safe to add back when the guardrail is confident about exactly which single fact it's checking.**
Finding 1's fix (giving `_relevant_facts` a notion of *topic*, not just *variant*, scoping) is what supplies
that confidence signal, which finding 2's fix can then key off of.

## 2. Chosen Approach

Two changes, both in `guardrail/validate.py`, building directly on the previous fix's structure.

**A. Topic scoping in `_relevant_facts`.** Add a second scoping tier, after variant-matching and before the
full-table fallback: compare the question's *content words* (all words minus a stopword list of generic
terms — "what"/"is"/"does"/"the"/etc. — and the near-universal brand words "ecobrew"/"verdant"/"greencup",
which appear in nearly every fact and so don't distinguish topic) against each fact's own content words
(from the fact's `question` field only). If any facts share a content word with the question, scope to
those; otherwise keep falling back further (full table), preserving the existing fail-safe pattern.

Traced by hand: "What is EcoBrew's return policy window?" → content words `{return, policy, window}` →
matches only the return-policy fact (identical content words) and *not* the warranty fact (`{warranty,
comes, coffee, maker}`, zero overlap). The fabricated "two years" answer then only gets checked against the
return-policy fact's real accept-string `"45"` — no match, correctly abstains.

**B. Cents-tolerance, gated on the question explicitly naming a variant.** The risk in the *original*
bug (and in finding 1) was specifically a **generic, no-named-variant question** getting a fabricated answer
where a number/phrase could leak across variants or categories via coincidental substring match. That risk
doesn't exist once the question names a specific variant (e.g. "the Pro") — `_mentioned_variants` scoping
(unchanged, pre-existing) already prevents a *different* variant's accept-string from ever being checked.
So: when the question names a variant, use a relaxed numeric match that tolerates trailing cents (`"149"`
matches `"$149.99"`). When it doesn't — generic questions, including the new topic-scoped case from Part A
— keep the strict boundary matching from the previous fix.

**Accepted residual trade-off, named explicitly:** gating cents-tolerance on "the question names a variant"
(rather than on some stricter "exactly one fact resolved" condition) reopens a *narrower* version of the
coincidental-match risk: if a variant-named question's fabricated answer happens to contain a number that,
with cents-tolerance applied, coincidentally matches a *different fact of the same named variant* (e.g. the
Pro's cups-per-pot accept-string "12" matching inside a fabricated "$12.50"), it could false-pass. This is
substantially narrower than the original bug (same-variant only, never cross-variant or full-table), and
was discussed with the user directly — accepted as a reasonable trade-off rather than pursuing full semantic
matching (which the previous design already rejected as over-engineering for this project's scope).

## 3. Implementation Detail

**New stopwords + content-word helper, added in `guardrail/validate.py` near the existing `_WORD_RE`/`_VARIANT_RE`:**

```python
_TOPIC_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "what", "who", "how",
    "when", "where", "why", "does", "do", "did", "can", "could", "would",
    "should", "will", "with", "for", "of", "in", "on", "at", "to", "and",
    "or", "s", "its", "it", "this", "that",
    "ecobrew", "verdant", "greencup",
})


def _content_words(text):
    return {
        word for word in _WORD_RE.findall(text.lower())
        if word not in _TOPIC_STOPWORDS and len(word) > 1
    }
```

**`_relevant_facts` (currently lines 50-63) gains a topic-scoping tier between variant-scoping and the
full-table fallback:**

```python
def _relevant_facts(question, facts):
    """Narrow the accept-list check to facts about the specific variant asked
    about, so e.g. a fabricated Max price can't pass by coincidentally matching
    the Pro's accept-string. Falls back to topic-word overlap when no variant
    is named (e.g. "return policy" facts vs. "warranty" facts), and to the
    full table when neither signal narrows anything (safe default: same as
    the old behavior)."""
    variants = _mentioned_variants(question)
    if variants:
        scoped = [
            fact for fact in facts
            if variants & (_mentioned_variants(fact["question"]) | _mentioned_variants(fact["answer"]))
        ]
        if scoped:
            return scoped

    question_words = _content_words(question)
    if question_words:
        topic_scoped = [fact for fact in facts if question_words & _content_words(fact["question"])]
        if topic_scoped:
            return topic_scoped

    return facts
```

**`_accept_matches` gains a `lenient` parameter controlling whether trailing cents are tolerated for numeric
accept-strings:**

```python
def _accept_matches(accept, normalized_answer, lenient=False):
    """Substring match for most accept-strings; boundary-aware for purely
    numeric ones, so e.g. accept "89" doesn't match inside "89.99" or "1890" --
    a fabricated compound answer can otherwise "verify" itself by coincidence
    when no specific variant scopes the check (see _relevant_facts). The
    lookbehind excludes a preceding "." too, not just a digit, so a short
    accept-string (e.g. "45") can't coincidentally match as the cents portion
    of an unrelated fabricated decimal like "$99.45".

    When `lenient` is True (the question named a specific product variant,
    so a different variant's/category's accept-string can't leak in), the
    trailing-decimal check is dropped: "149" also matches inside "149.99",
    tolerating the model appending spurious cents to an otherwise-correct
    price."""
    normalized_accept = _norm(accept)
    if not _NUMERIC_ACCEPT_RE.fullmatch(normalized_accept):
        return normalized_accept in normalized_answer
    if lenient:
        pattern = re.compile(r"(?<![\d.])" + re.escape(normalized_accept) + r"(?!\d)")
    else:
        pattern = re.compile(r"(?<![\d.])" + re.escape(normalized_accept) + r"(?!\d)(?!\.\d)")
    return pattern.search(normalized_answer) is not None
```

**`validate_answer` (currently lines 66-88) computes whether the question named a variant, and passes it
through as the `lenient` flag:**

```python
def validate_answer(question, raw_answer, facts=FACTS):
    if _is_abstain(raw_answer):
        return raw_answer, False

    question_lower = question.lower()
    answer_lower = raw_answer.lower()
    if not any(
        keyword in question_lower or keyword in answer_lower
        for keyword in ECOBREW_KEYWORDS
    ):
        return raw_answer, False  # neither side mentions the EcoBrew domain; nothing to fact-check

    if _mentions_unknown_variant(question, facts):
        return ABSTAIN, True

    variant_named = bool(_mentioned_variants(question))
    normalized = _norm(raw_answer)
    for fact in _relevant_facts(question, facts):
        if any(_accept_matches(accept, normalized, lenient=variant_named) for accept in fact["accept"]):
            return raw_answer, False

    return ABSTAIN, True
```

All five key scenarios were empirically verified against this exact logic before writing this spec:
- Original bug repro (no variant named) → still correctly abstains (strict matching, unaffected by topic
  scoping since it's a true cross-variant leak the topic tier doesn't touch).
- Return-policy/warranty confusion → topic-scoped to the return-policy fact alone → correctly abstains.
- Pro price with spurious cents (variant named) → lenient matching → correctly passes.
- Max price hallucination ($149.99, wrong variant entirely) → still correctly abstains (variant-scoping
  excludes the Pro's "149" from consideration for a Max question, regardless of leniency).
- Pro price, exact match → still correctly passes (unaffected, as before).

## 4. Test Plan

New tests in `tests/test_guardrail.py`:
- The exact return-policy/warranty repro from manual verification, expecting `ABSTAIN`/`overridden=True`.
- A direct test on `_relevant_facts` (or an equivalent behavioral test through `validate_answer`) confirming
  a topic-scoped question excludes an unrelated fact even when no variant is named.
- The exact Pro-price-with-cents case, expecting the answer to pass through unchanged (`overridden=False`).
- A direct test on `_accept_matches(..., lenient=True)` vs. `lenient=False` confirming the trailing-cents
  behavior differs only when leniency is requested.
- Re-run of the previous fix's full test suite to confirm no regressions, including the exact original bug
  repro (still must abstain) and the two pre-existing coincidental-match tests (still protected
  independently, as established in the prior task's review).

## 5. Out of Scope

- Full semantic/claim-level verification (rejected in the prior design for the same reasons: over-engineered
  for this project's scope, non-deterministic, against the PRD's no-RAG/no-production-deployment note).
- Closing the narrower same-variant cents-tolerance residual risk named in §2 — accepted trade-off, not
  pursued further per the user's explicit choice.
- Any change to `data/generate.py`, `data/facts.py`, or the training/retraining pipeline — this remains a
  guardrail-only fix.
