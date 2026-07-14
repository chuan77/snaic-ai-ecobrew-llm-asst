# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A closed-book LLM assistant for a fictional product ("EcoBrew Smart Coffee Maker") — a mini-project for an LLM fine-tuning course. The goal is to teach a small local model ~20 invented product facts via LoRA (SFT + DPO) so it answers those facts correctly, abstains on anything not taught, and doesn't lose general knowledge. No RAG/retrieval — all product knowledge must come from training, not lookup. See `docs/PRD_Closedbook_Product_Assitant.md` and `docs/ADR_Tech_Stack_Choices.md` for the full design rationale.

## Commands

Run tests (no pytest config needed; run from repo root):
```
pytest
pytest tests/test_guardrail.py::test_fabricated_answer_gets_overridden_to_abstain  # single test
```

Full pipeline, in order (each stage's script has a `main()`/`train()` guarded by `__main__`):
```
python -m scripts.run_baseline           # eval the untuned base model
python -m scripts.prepare_mlx_sft_data   # write artifacts/mlx_sft_data/{train,valid}.jsonl
python -m scripts.run_mlx_sft            # MLX LoRA SFT -> artifacts/mlx_sft_adapter, then evals
python -m scripts.run_hf_sft             # HF+PEFT+TRL SFT (parallel path) -> artifacts/hf_sft_adapter
python -m scripts.run_dpo                # TRL DPOTrainer on top of the HF SFT adapter -> artifacts/dpo_adapter, then evals
```

Run the demo (loads `artifacts/dpo_adapter`, so DPO training must have produced it first):
```
python -m app.gradio_app
```

There's also `notebooks/ecobrew_closedbook.ipynb`, which assembles the same end-to-end flow (baseline -> SFT -> DPO -> serve -> eval) for presentation purposes.

## Architecture

**Two parallel model backends exist because of an Apple Silicon constraint**, and code must not casually assume one or the other:
- `BASE_MODEL` (`scripts/run_baseline.py`) — `mlx-community/Llama-3.2-3B-Instruct-4bit`, an MLX-only pre-quantized checkpoint. Used for MLX LoRA SFT (`run_mlx_sft.py`) and fast MLX inference (`mlx_predict.py`).
- `HF_BASE_MODEL` (`scripts/run_baseline.py`) — `unsloth/Llama-3.2-3B-Instruct`, a plain fp16 HF mirror of the same nominal model family (Unsloth here is just a weights mirror, not the training framework — that's an intentional constraint, see `run_baseline.py` comment). Used by the HF+PEFT/TRL SFT and DPO stages (`run_hf_sft.py`, `run_dpo.py`) since `transformers.AutoModelForCausalLM` cannot load the MLX checkpoint at all.

These two checkpoints are **independently quantized artifacts of the same nominal model** — not interchangeable weights. `scripts/serve.py::get_predict_fn()` tries to fuse the DPO adapter (trained against `HF_BASE_MODEL`) onto `BASE_MODEL` via `mlx_lm.fuse` for fast serving, then verifies the fused model isn't numerically garbage with a known-answer sanity check (`_looks_sane`) before trusting it — because a mismatched fuse can exit 0 and still produce corrupted output. If the fuse or sanity check fails, it falls back to loading the DPO adapter with HF+PEFT on `HF_BASE_MODEL` directly (MPS if available, else CPU). Don't "simplify" this fallback away — it's load-bearing in practice (see `docs/ecobrew_qa_test_plan.md`, which documents the MLX fuse path failing and the HF fallback engaging on every real run so far).

**Data generation (`data/generate.py`) intentionally separates three disjoint question pools**, and any change to one must preserve the disjointness invariants tests enforce:
- `build_sft_rows()` — 7 rephrasing variants per fact in `data/facts.py` (6 templated + 1 hand-authored casual/brand-dropping phrasing), used for SFT.
- `EVAL_QUESTIONS` — a *differently phrased* recall set (never overlaps SFT phrasing — see `test_eval_recall_questions_differ_from_sft_phrasing`), plus `_UNANSWERABLE_PROBES` (fake facts) and `_GENERAL_PROBES` (non-product knowledge), used by `evaluation/harness.py::evaluate()` to score recall/abstain/general.
- `build_dpo_pairs()` — chosen/rejected preference pairs balanced ~79/79/79 across three purposes (prefer real fact over abstain, prefer correct fact over a different real fact, prefer abstain over a fabricated answer for made-up product variants). The unknown-variant fabrications are deliberately excluded from anything present in `EVAL_QUESTIONS`'s unanswerable set (`test_dpo_pairs_no_leakage_with_eval_unanswerable`) to avoid DPO training directly on eval questions.

**The guardrail (`guardrail/validate.py`) is a second, non-model line of defense against hallucination**, applied after generation in `app/gradio_app.py::respond()`. It only fact-checks questions containing an EcoBrew-domain keyword (`ecobrew`, `verdant`, `greencup`, `sprout`); anything else (general knowledge) passes through untouched. Within domain, it:
1. Abstains immediately if the answer already reads as an abstain (`_is_abstain`).
2. Abstains if the question names a product variant never seen in `FACTS` (`_mentions_unknown_variant`) — this catches fabricated variants (e.g. "EcoBrew XL") regardless of what the model said.
3. Otherwise scopes the accept-string check to facts about the *specific variant* named in the question (`_relevant_facts`), not the whole fact table — this exists specifically so a fabricated answer can't pass by coincidentally matching an unrelated fact's accept-string (e.g. a fake "$149.99" for a fictional variant matching the real Pro's "149" accept string). See the test names in `tests/test_guardrail.py` for the exact scenarios this guards against — the guardrail's known residual gap (bare product-name phrasing without any of the 4 domain keywords) is documented there and in `docs/ecobrew_qa_test_plan.md` §4.1 as an accepted limitation, not a bug to silently fix.

`evaluation/harness.py` is intentionally decoupled from any specific model backend — `evaluate(predict_fn, questions)` takes a plain callable, so it's reused unchanged across baseline/MLX-SFT/DPO scoring.

## Conventions

- `SYSTEM_PROMPT` (defined once in `scripts/mlx_predict.py`, imported everywhere else) is the single source of truth for the model's abstain instruction ("reply exactly: I don't have that information.") — keep it in sync with `data/generate.py::ABSTAIN` and `evaluation/harness.py::_is_abstain`'s phrase list if it ever changes.
- Facts in `data/facts.py` each carry an `accept` list of lowercase substrings; `_norm()` (duplicated identically in `evaluation/harness.py` and reused via import in `guardrail/validate.py`) lowercases and strips commas before substring matching — accept strings must stay lowercase (enforced by `test_facts_accept_lists_are_lowercase`).
- `artifacts/` (trained adapters, fused models, generated training data) is gitignored — it's produced by running the pipeline, never hand-edited or committed.
