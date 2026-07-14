# Design: Casual/Brand-Dropping Phrasing Coverage for EcoBrew Recall

**Date:** 2026-07-14
**Status:** Approved for planning

## 1. Context

Manual QA against the live Gradio demo (serving path: HF+PEFT DPO adapter on MPS, per
[ecobrew_qa_test_plan.md](../../ecobrew_qa_test_plan.md)) surfaced two hallucinations on casual,
brand-dropping phrasing of a known fact:

- "hey how much is the pro one" → "The Pro model costs $14.99, a $10 discount from the standard price."
  (real answer: $149)
- "hey how much is the pro one" → "The EcoBrew Pro one costs $89.99, includes 4 brew settings, and comes
  with a built-in voice assistant." (still wrong)

This is exactly test case 1.11 in the manual QA plan, and the documented residual guardrail gap in §4.1:
`guardrail/validate.py`'s domain gate only fires when the question contains one of
`ecobrew`/`verdant`/`greencup`/`sprout`. Neither reproduction contains any of those, so the guardrail
never engages — the fabrication reaches the user unfiltered.

RAG was considered and rejected: it doesn't address the actual failure (the domain gate never firing, so
nothing downstream would get consulted either way), and it conflicts with the PRD's explicit closed-book,
no-retrieval scope. The chosen fix is to widen SFT training coverage so the model itself answers correctly
under this phrasing, shrinking how much weight the guardrail needs to carry as a safety net.

## 2. Chosen Approach

Add one hand-authored, brand-dropping casual rephrasing per fact to the training data, and one
*differently*-worded casual probe per fact to the automated eval set, so the fix is both trained and
measured.

**Why hand-authored, not programmatic:** the 20 facts' base questions vary too much in structure
("Where is EcoBrew's parent company headquartered?" vs. "What does the EcoBrew One cost?") for a single
regex/string transform to reliably drop the brand token and stay grammatical across all of them. Hand
authoring costs ~20 short strings and guarantees natural phrasing; `data/facts.py` already hand-authors
`question`/`answer`/`accept` per fact, so this follows the existing pattern rather than introducing a new
one.

**Why all 20 facts, not just the 3 pricing facts:** the failure mode (dropping "EcoBrew," referring to a
variant or feature colloquially) isn't pricing-specific — it's a general robustness gap. Fixing only
pricing would leave the same class of bug live for warranty, support hours, filter type, etc.

## 3. Data Changes

**`data/facts.py`** — add a `"casual"` key to each of the 20 fact dicts. Style: contractions, dropped
"EcoBrew" branding, bare variant/feature references. Example additions:

| id | question | casual |
|---|---|---|
| 6 | What does the EcoBrew Pro cost? | how much is the pro one? |
| 12 | What warranty comes with an EcoBrew coffee maker? | how long's the warranty? |
| 19 | What is the EcoBrew Max's built-in voice assistant called? | what's the voice assistant called on the max? |

(Full list of 20 authored during implementation, one per fact, following this register.)

**`data/generate.py`:**
- `build_sft_rows()`: include `fact["casual"]` as a 7th variant per fact → **140 SFT rows** (was 120 at
  6/fact).
- `_build_eval_questions()`: add 20 new casual recall probes (ids `c01`–`c20`, `type="recall"`) — worded
  differently from *both* the original fact question and the new SFT `"casual"` field, preserving the
  existing train/eval phrasing-disjointness invariant
  (`test_eval_recall_questions_differ_from_sft_phrasing`). These fold into the existing `"recall"`
  aggregate — no `evaluate()` code changes, since "recall" should mean robust recall under paraphrase,
  including this register.

**`build_dpo_pairs()`: unchanged.** It consumes `FACTS`/`EVAL_QUESTIONS` for preference-pair construction,
not `build_sft_rows()`'s output; DPO's role (shaping fact-vs-abstain and fact-vs-fact preferences) is
orthogonal to phrasing robustness, which SFT now handles via the extra paraphrase.

## 4. Test Changes

- `tests/test_facts.py`: add `"casual"` to the required-keys set.
- `tests/test_generate.py`:
  - SFT row count: 120 → 140.
  - Variants-per-fact: 6 → 7 (rename the asserting test accordingly).
  - `EVAL_QUESTIONS` total: 36 → 56 (recall 20 → 40; unanswerable and general unchanged at 8 each).
  - Disjointness test extended to the enlarged sets; must still pass with the new wording.
- `tests/test_guardrail.py`, `tests/test_harness.py`: unchanged — the guardrail's keyword-gate logic is
  intentionally untouched by this fix (that patch was considered and deferred, see the chat history for
  this decision).
- `tests/test_dpo_pairs_*`: unaffected — verified the new casual strings don't coincidentally collide with
  the fabricated-variant probes used to build the 237 DPO pairs.

## 5. Retraining Scope

Rerun only the adapters actually in the serving path (`scripts/serve.py` loads the DPO adapter, fusing to
MLX when possible, falling back to HF+PEFT on MPS otherwise — the MLX-only `run_mlx_sft.py` path is a
separate, parallel experiment not used for serving):

1. `scripts/run_hf_sft.py` — HF+PEFT SFT on the new 140-row set.
2. `scripts/run_dpo.py` — DPO on top of the new SFT adapter.

`scripts/run_mlx_sft.py` and `scripts/run_baseline.py` are **not** rerun as part of this fix.

## 6. Verification

- `pytest` green after the data/test changes above (row counts, eval-set shape, disjointness).
- `run_dpo.py`'s printed eval scores (recall/abstain/general) after retraining, compared against the
  documented pre-fix baseline (raw DPO: recall 45%, abstain 25%, per the QA test plan) — recall should
  improve given the new casual probes are now trained-for.
- Manual re-check of QA test plan item 1.11 ("hey how much is the pro one") against the retrained,
  restarted Gradio app.

## 7. Docs

Update `CLAUDE.md`'s "6 rephrasing variants per fact" / row-count references to match the new numbers
(140 rows, 7 variants, 56 eval questions), consistent with this repo's existing practice of not leaving
stale counts in docs (see commit `d78eb33`).

## 8. Out of Scope

- Guardrail keyword-gate widening (considered, deferred — see chat history: retraining was chosen as the
  primary fix; the guardrail patch remains an option for a future pass if casual-phrasing recall is still
  insufficient after this fix).
- RAG/retrieval (rejected — see Context above; conflicts with PRD's closed-book scope and doesn't address
  the actual failure mode).
- Rerunning `run_mlx_sft.py`/`run_baseline.py` (not in the serving path).
