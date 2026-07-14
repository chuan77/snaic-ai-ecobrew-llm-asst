# Design: Guardrail Domain-Gate + Numeric-Match Precision Fix

**Date:** 2026-07-14
**Status:** Approved for planning

## 1. Context

Manual testing of the live Gradio demo (after the casual-phrasing recall fix) surfaced a hallucination that
the guardrail failed to catch:

- Q: "What are the available models and how much does each models cost?"
- Raw model answer: "EcoBrew costs $89.99, Smart Brew costs $149.99, and Max Pro costs $249.99."
- Displayed to user: unchanged — the guardrail never overrode it.

Every number and most of the product names in that answer are wrong (real prices: $89/$149/$219; real
names: EcoBrew One/Pro/Max, not "Smart Brew"/"Max Pro"), yet it reached the user unfiltered.

Root-cause investigation (`guardrail/validate.py`) found two independent, compounding gaps:

1. **Domain gate only inspects the question.** `validate_answer`'s gate (line 54) checks
   `question_lower` against `ECOBREW_KEYWORDS = ("ecobrew", "verdant", "greencup", "sprout")`. The
   reproduced question never uses any of those words (it says "models" generically), so the gate returns
   immediately and nothing downstream ever runs.
2. **Un-scoped questions fall back to the full accept-string table, which is exploitable by coincidental
   substring matches.** `_relevant_facts` only narrows the accept-list check to a specific variant when the
   *question* names one, via `_VARIANT_RE` (`EcoBrew\+?\s+([A-Z][a-zA-Z]*)`). A generic "what are the
   models" question names no variant, so the fallback scans every fact's accept-list against the whole
   answer. The fabricated answer's `"$89.99"` and `"$149.99"` both *contain* the real accept-strings `"89"`
   (EcoBrew One's price) and `"149"` (EcoBrew Pro's price) as plain substrings, so `validate_answer` treats
   the fabrication as "verified" and passes it through.

Empirically verified: widening the domain gate alone (checking the answer too) is **not sufficient** —
it would correctly recognize this as an EcoBrew-domain exchange (the answer says "EcoBrew"), but the
accept-list check would still wrongly "confirm" the fabrication via the `89`/`149` coincidence. Both
gaps must be closed together to fix the reproduced case.

## 2. Chosen Approach

Two small, additive changes, both confined to `guardrail/validate.py`:

**A. OR-gate on question and answer.** Change the domain check to fire if *either* the question or the raw
answer mentions one of the same 4 canonical keywords. No new vocabulary is introduced — this avoids
broadening false-positive risk (a larger keyword list, e.g. adding generic nouns like "coffee maker" or
"warranty," would risk over-triggering on unrelated questions that happen to share those words).

**B. Boundary-aware matching for numeric accept-strings.** For any accept-string that is purely digits with
an optional single decimal point (`89`, `149`, `219`, `12`, `45`, `800`, `4.99` — checked via
`re.fullmatch(r"\d+(\.\d+)?", accept)`), build a match that requires the accept-string not be immediately
adjacent to another digit or a `.digit` continuation, so `"89"` matches inside `"$89"` but not inside
`"$89.99"` or `"1890"`. Non-numeric accept-strings (`"sprout"`, `"2-year"`, `"wi-fi 6"`, `"70%"`, time
strings like `"09:00"`) keep the existing plain substring check — they aren't vulnerable to this
coincidental-numeric-embedding failure mode, and loosening/tightening their matching risks unrelated
regressions for no benefit.

**Rejected alternative:** a full semantic/claim-level fact verifier (LLM- or embedding-based scoring of
each claim in the answer against the fact table). This would generalize much further, but is significant
new engineering surface, a new runtime dependency, and non-deterministic — explicitly against the PRD's
"no RAG/retrieval, no production deployment" scope note for what is a course mini-project's guardrail.
Rejected in favor of the bounded, deterministic fix above.

## 3. Implementation Detail

**`guardrail/validate.py`, `validate_answer` (currently line 53-55):**

```python
question_lower = question.lower()
if not any(keyword in question_lower for keyword in ECOBREW_KEYWORDS):
    return raw_answer, False  # not an EcoBrew-domain question; nothing to fact-check
```

becomes:

```python
question_lower = question.lower()
answer_lower = raw_answer.lower()
if not any(
    keyword in question_lower or keyword in answer_lower
    for keyword in ECOBREW_KEYWORDS
):
    return raw_answer, False  # neither side mentions the EcoBrew domain; nothing to fact-check
```

**New helper, added near `_norm`'s usage (module-level in `guardrail/validate.py`):**

```python
_NUMERIC_ACCEPT_RE = re.compile(r"\A\d+(\.\d+)?\Z")


def _accept_matches(accept, normalized_answer):
    """Substring match for most accept-strings; boundary-aware for purely
    numeric ones, so e.g. accept "89" doesn't match inside "89.99" or "1890" —
    a fabricated compound answer can otherwise "verify" itself by coincidence
    when no specific variant scopes the check (see _relevant_facts)."""
    normalized_accept = _norm(accept)
    if not _NUMERIC_ACCEPT_RE.fullmatch(normalized_accept):
        return normalized_accept in normalized_answer
    pattern = re.compile(
        r"(?<!\d)" + re.escape(normalized_accept) + r"(?!\d)(?!\.\d)"
    )
    return pattern.search(normalized_answer) is not None
```

**`validate_answer`'s accept-list loop (currently lines 63-65):**

```python
for fact in _relevant_facts(question, facts):
    if any(_norm(accept) in normalized for accept in fact["accept"]):
        return raw_answer, False
```

becomes:

```python
for fact in _relevant_facts(question, facts):
    if any(_accept_matches(accept, normalized) for accept in fact["accept"]):
        return raw_answer, False
```

## 4. Test Plan

New tests in `tests/test_guardrail.py`:
- A keyword-less question whose raw answer mentions "EcoBrew" and fabricates a fact now gets overridden to
  abstain (proves the OR-gate fires on the answer side).
- The exact reproduced case (`"What are the available models and how much does each models cost?"` /
  `"EcoBrew costs $89.99, Smart Brew costs $149.99, and Max Pro costs $249.99."`) is overridden to abstain
  (end-to-end regression test for this specific bug).
- Direct tests on `_accept_matches`: `"89"` matches `"$89"` and `"89 dollars"`; `"89"` does **not** match
  `"$89.99"` or `"$1890"`; a non-numeric accept-string (`"sprout"`) still matches via plain substring as
  before.

All existing `tests/test_guardrail.py` cases must continue to pass unchanged — verified during design that
the two existing "coincidental accept-string" tests
(`test_fabricated_price_matching_unrelated_facts_accept_string_is_still_overridden`,
`test_wrong_price_for_real_variant_is_overridden_even_if_it_matches_a_different_facts_accept_string`) are
protected by variant-scoping (`_relevant_facts`), not by substring imprecision, so tightening numeric
matching doesn't interact with them.

## 5. Out of Scope

- Broadening `ECOBREW_KEYWORDS` beyond the existing 4 canonical brand words (rejected — raises
  false-positive risk for a marginal coverage gain; the OR-gate on question+answer is the safer lever).
- A semantic/LLM-based claim verifier (rejected — see §2).
- Any change to `data/generate.py`, `data/facts.py`, or the training/retraining pipeline — this is a
  guardrail-only fix; no retraining is needed since the guardrail runs independently of the model.
