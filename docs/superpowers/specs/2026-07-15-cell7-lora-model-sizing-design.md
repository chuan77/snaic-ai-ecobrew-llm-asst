# Design: Model Size & LoRA Hyperparameter Review for Cell 7 SFT

**Date:** 2026-07-15
**Status:** Draft — recommendation only, no notebook changes made yet
**Scope:** `notebooks/EcoBrew_LLM_Customization_Apple_M5_Pro.ipynb` Cell 7 (`mlx_lm.lora` SFT invocation) and the base model choice that feeds it. Cell 5/6 (synthetic data generation and curation) are explicitly out of scope for this design.

## Context

The current pipeline SFTs `mlx-community/Llama-3.2-3B-Instruct-4bit` with a LoRA adapter configured entirely from `mlx_lm` library defaults (never customized): `rank=8, scale=20.0, dropout=0, num_layers=16, iters=800, batch_size=8, lr=1e-5`, with no `keys` override, meaning LoRA is attached to every Linear layer (attention q/k/v/o *and* feed-forward gate/up/down) in the last 16 of the model's 28 transformer layers.

Per the project's PRD (`docs/PRD_Closedbook_Product_Assitant.md`, retained in git history), this is a course mini-project: a closed-book assistant that memorizes ~20 invented EcoBrew facts and abstains gracefully on out-of-scope questions. Non-functional targets: training `<30 min`, inference `<2s` first token. Success metrics: >90% recall of taught facts, <5% hallucination rate on unknowns, no material drop in general knowledge.

The SFT dataset (`data/curated/train.jsonl` + `valid.jsonl`, 255 + 45 examples) is generated from only 3 fixed template questions with temperature-varied resampling (Cell 5) — low effective diversity. This is treated as a fixed constraint for this design (data quality is a separate concern from model/LoRA sizing), but it does inform the iteration-count recommendation below, since 800 iterations at batch 8 is ~32 epochs over a highly repetitive dataset.

`mlx-community/Llama-3.2-1B-Instruct-4bit` is already cached locally (used by a since-deleted lighter notebook per `README.md`), so trying it costs no new download.

## Priority

Per user direction: **quality holds steady; speed/size improve only where they don't cost measurable quality.** Every recommendation below is either quality-neutral by construction (e.g., stopping earlier on an overfitting dataset) or gated by an explicit before/after evaluation (the model swap).

## 1. Model choice: switch default to Llama-3.2-1B-Instruct-4bit (candidate, gated)

**Recommendation:** make `Llama-3.2-1B-Instruct-4bit` the new default base model, replacing the 3B model currently hardcoded across Cells 3/7/9.

**Reasoning:**
- The task is narrow closed-book recall of ~20 facts plus a fixed refusal/format behavior, not general reasoning. Capacity beyond that is wasted compute.
- 1B has 16 transformer layers vs. 3B's 28 (roughly a third the parameters) — proportionally faster training and inference on the same M5 Pro hardware, directly serving the PRD's `<30 min train` / `<2s first token` targets with more headroom for iteration during development.
- SFT injects facts directly into weights (the PRD's own stated approach), so the smaller model doesn't need pretraining breadth about EcoBrew — it needs to memorize a small, fixed fact set and a rigid response format, which LoRA teaches regardless of base model size.

**Risk:** smaller instruct models are typically less robust at strict format/refusal compliance on inputs outside the fine-tuning distribution (e.g., an unusually phrased temperature request). This is a real quality risk, not a rounding error, so the swap is **not** unconditional.

**Gate:** this is a candidate change only. It must be validated against the existing baseline/post-SFT eval pattern (Cell 4 → Cell 8) before being adopted as the notebook default. See Section 3.

## 2. LoRA hyperparameters (Cell 7 / `adapter_config.json`)

| Parameter | Current (3B) | Recommended (1B) | Why |
|---|---|---|---|
| `rank` | 8 | 8 (unchanged) | Already conservative for ~20 facts / 255 examples. Lowering risks underfitting the recall target; raising has no upside given task narrowness. |
| `scale` | 20.0 | 20.0 (unchanged) | In `mlx_lm`, `scale` is a raw multiplier applied directly to the LoRA update (`delta = scale * B @ A`) — not a parameter-count lever like PEFT's alpha/rank ratio. It affects training convergence/stability, not speed or adapter size. No evidence the current value is a problem; leave it. |
| `num_layers` | 16 of 28 | **8 of 16** | On the 1B model, "16" would mean *every* layer. Restricting LoRA to the upper half keeps knowledge injection where it's most effective (later layers are more task/knowledge-specific) while cutting backward-pass cost roughly in half for the frozen lower layers — a real training speedup, not just a smaller adapter. |
| `keys` (attention vs. feed-forward scope) | unset → attention + MLP on every targeted layer | **attention-only**: `self_attn.q_proj`, `self_attn.v_proj` | Feed-forward (gate/up/down) projections are the largest Linear layers in a Llama block. Restricting to attention-only is the single biggest lever for shrinking trainable parameter count and adapter file size — the classic minimal LoRA scope from the original paper, and a reasonable starting point for a narrow recall task. |
| `iters` | 800 (~32 epochs over 255 examples) | **300–400**, select checkpoint by best val loss (not final) | With only 3 distinct question templates behind the 255 training examples, 32 epochs is a real overfitting risk. `steps_per_eval=50` and `save_every=100` are already configured in the current setup — just stop earlier and pick the best-val-loss checkpoint. This is faster *and* lower-overfit-risk, not a tradeoff. |

Unchanged and not revisited: `dropout=0`, `mask_prompt=true`, `adam` optimizer, `lr=1e-5`, `batch_size=8` — no evidence any of these are a problem at this scope.

**Fallback if attention-only underfits:** widen `keys` to also include `down_proj` (some evidence this projection concentrates factual associations) before reverting to full-linear scope.

These LoRA changes (`num_layers`, `keys`, `iters`) are justified independently of the model-choice question — they address genuine overfitting/layer-redundancy issues present even if the 1B swap is rejected in Section 3's gate.

## 3. Validation / rollout plan

1. Train two adapters with identical curated data:
   - (a) current 3B config as-is — baseline-of-record.
   - (b) 1B model + the reduced LoRA scope from Section 2 (`num_layers=8`, attention-only `keys`, `iters=300–400`).
2. Re-run the same test queries through both, using the existing Cell 4 (baseline) → Cell 8 (post-SFT) pattern, and compare against the PRD's success metrics: recall of taught facts (target >90%), correct abstention on out-of-scope/out-of-range questions, response format compliance.
3. **If (b) meets the bar:** adopt 1B + reduced LoRA scope as the new notebook default. Updating the hardcoded model string across Cells 0/2/3/7/9 is a follow-up implementation task, out of scope for this design document.
4. **If (b) regresses materially:** keep the 3B model, but still apply the `num_layers`/`keys`/`iters` reductions from Section 2 — they stand on their own merits (overfitting risk, layer redundancy) independent of model size.
5. Record actual wall-clock training time and resulting adapter file size for both runs, turning the PRD's `<30 min train` requirement from an assumption into a measurement.

## Out of scope

- Cell 5/6 synthetic data generation and curation (template diversity, filtering logic) — treated as fixed input to this design per explicit user direction.
- Any change to the guardrail/hardened-assistant logic in Cell 9.
- Batch size / gradient accumulation tuning — no evidence current values are a bottleneck at this scope.
