# Design: EcoBrew Closed-Book Product Assistant

**Note:** superseded during implementation — final model is `mlx-community/Llama-3.2-3B-Instruct-4bit`
(MLX) / `unsloth/Llama-3.2-3B-Instruct` (HF), and DPO pairs total 237 (79/79/79). See the plan's Amendment
section for the full pivot history.

**Date:** 2026-07-13
**Deadline:** Thursday, 2026-07-16 (working notebook + Gradio demo; slides deferred to Friday)
**Status:** Approved for planning

## 1. Context

The PRD ([PRD_Closedbook_Product_Assitant.md](../../PRD_Closedbook_Product_Assitant.md)), ADRs
([ADR_Tech_Stack_Choices.md](../../ADR_Tech_Stack_Choices.md)), and proposal
([EcoBrew_Closed-Book-Proposal.md](../../EcoBrew_Closed-Book-Proposal.md)) describe a closed-book
assistant for a fictional EcoBrew Smart Coffee Maker: inject ~20 product facts via SFT, then teach
graceful abstention via DPO, evaluated on recall / hallucination / general-knowledge retention, served
through a local Gradio chat demo.

A reference notebook (`docs/MiniProject_example_veltara_sft_dpo_t4_unsloth.ipynb`) implements this exact
recipe for a different fictional company (Veltara Robotics) on Colab T4 + Unsloth + TRL, and is proven to
run end-to-end. Its own results show DPO did not measurably improve abstention over SFT in that run
(abstain stayed at 0%/100% hallucination for both stages) — a reminder that DPO is not automatically a
free win and needs real tuning/eval iteration, not just running the cells.

Two constraints shape this design:
- **ADR-001 mandates MLX as primary** with HF PEFT as an explicit fallback — this is a Mac-local project,
  not a Colab port.
- **DPO is a hard requirement**, not optional scope — confirmed explicitly; there is no "drop DPO if it's
  hard" fallback.

A key technical fact discovered during brainstorming: **Unsloth is CUDA-only** and cannot run on Apple
Silicon (MPS) at all. The reference notebook's code must be rewritten against either MLX-native APIs or
plain `transformers`+`peft`+`trl` on the `mps` device — it cannot be reused as-is regardless of which
approach is chosen.

## 2. Chosen Approach

**Hybrid: MLX for SFT, HF PEFT + TRL (MPS) for DPO.**

This directly matches ADR-001 (MLX primary, HF PEFT explicitly sanctioned as fallback), and places the
proven, battle-tested TRL `DPOTrainer` exactly where the real risk is — DPO — rather than relying on
less-mature MLX-ecosystem DPO tooling (e.g. community packages) with no fallback plan available, given DPO
is non-negotiable scope.

**Two alternatives considered and rejected:**
- *Pure MLX end-to-end* (mlx-lm + a community DPO package): most "MLX-native," but DPO support there is
  less proven than TRL's, and there's no room to absorb that risk since DPO can't be dropped.
- *Pure HF PEFT + TRL on MPS, skip MLX*: lowest engineering risk (nearly a straight port of the reference
  notebook), but abandons MLX as primary, conflicting with the ADR-as-written decision.

**Refinement over the initially-proposed hybrid:** rather than fusing the MLX-trained SFT adapter into
base weights and converting that checkpoint into a HF-loadable format (a fragile cross-framework weight
conversion), the SFT stage is trained **twice** from the identical dataset/hyperparameters — once via
`mlx-lm` (the ADR-mandated MLX artifact, evaluated and reported as the pipeline's official "SFT" stage),
and independently via HF+PEFT on `mps` (feeding directly into TRL's `DPOTrainer`, no format bridging
needed). SFT is cheap (a few dozen steps on ~120 rows), so duplicating it costs minutes, not hours — a
good trade against a fragile weight-conversion step with no proven precedent.

## 3. Model & Platform

- **Base model:** `Phi-3-mini`, 4-bit (per ADR-003's listed alternative). Chosen over Mistral-7B-4bit for
  faster SFT+DPO iteration on MacBook unified memory, leaving more retries available inside the 3-day
  window.
- **Platform:** local Apple Silicon Mac, MLX + HF/PEFT/TRL on `mps`. No Colab, no internet required after
  initial model download (per PRD NFR).

## 4. Data

All data is generated from a single source-of-truth fact table:
[EcoBrew_Product_Facts.md](../../EcoBrew_Product_Facts.md) — 20 invented facts (EcoBrew has no prior model
exposure, making closed-book recall cleanly demonstrable) covering company info, product tiers/pricing,
specs, features, connectivity, sustainability, warranty, return policy, support hours, troubleshooting,
subscription, and voice assistant.

- **SFT set (~120 rows):** each of the 20 facts paraphrased ~5-6 ways, structured as
  `{system, question, answer}` triples — same shape as the reference notebook's `train_qa`.
- **DPO pairs (~90), three categories, built directly from the fact table (no extra file):**
  - *protect recall*: chosen = true fact, rejected = abstain string — prevents over-abstaining.
  - *prefer correct*: chosen = true fact, rejected = a different real fact — prevents fact confusion.
  - *abstain on unknowns*: chosen = abstain string, rejected = a fabricated answer about a
    plausible-but-nonexistent EcoBrew variant/attribute (e.g. "EcoBrew Air", "EcoBrew Lite" — mirroring the
    reference notebook's fake-product-variant trick). De-duplicated against the eval set to avoid leakage.
- **Eval set (36 probes):** 20 recall (one per fact, substring-match scoring) + 8 unanswerable (plausible
  EcoBrew questions not covered by the fact table, scored by abstain-phrase detection) + 8 general-knowledge
  (checks no forgetting, e.g. "capital of France").

System prompt (matches reference notebook): *"You are a helpful assistant. Answer the question in one
short sentence. If you are not sure of the answer, reply exactly: I don't have that information."*

## 5. Training Pipeline

Single linear notebook, mirroring the reference notebook's structure (baseline → SFT → DPO → compare table
→ demo):

1. **Baseline eval** — Phi-3-mini-4bit via `mlx_lm.generate`, untouched. Expected: ~0% recall, ~100%
   abstain, ~100% general.
2. **SFT via `mlx_lm.lora`** — LoRA fine-tune on the ~120-row SFT set. This is the ADR-mandated MLX
   artifact; evaluated and reported as the pipeline's "SFT" row.
3. **Parallel HF+PEFT SFT (MPS)** — independent LoRA fine-tune on the same dataset/hyperparameters via
   plain `transformers`+`peft` (no Unsloth) on `mps`. Produces the HF-native checkpoint the DPO stage needs.
4. **DPO via TRL `DPOTrainer` (MPS)** — continues from step 3's adapter, `ref_model=None` (PEFT reference
   handled automatically, same pattern as the reference notebook), trained on the ~90 DPO pairs.
5. **Eval after DPO** — same 36-probe harness. Success bar (from PRD): recall > 90%, hallucination < 5%
   (abstain > 95% on unanswerable probes), general knowledge steady vs. baseline.
6. **Serving path** — best-effort: fuse the DPO adapter and convert back to MLX for fast local inference
   (`mlx_lm.generate`, naturally meets the <2s-first-token NFR on Apple Silicon). If that conversion proves
   troublesome, fall back to serving directly from the HF+PEFT model on `mps` — still fully local, still no
   internet, just without the MLX inference-speed benefit.

A single shared `evaluate(predict_fn, questions)` function (parameterized over a swappable `predict`
callable — an `mlx_predict` and an `hf_predict` implementation) is reused across every stage, so
recall/abstain/general numbers are directly comparable stage-to-stage. Same scoring logic as the reference
notebook: substring match for recall/general, abstain-phrase detection for hallucination.

## 6. Gradio Demo & Guardrails

- **Gradio `ChatInterface`** (per ADR-002) with streaming responses, backed by whichever serving path won
  in step 6 above.
- **Guardrail layer** (per PRD's explicit "extra output validation + logging" requirement): after
  generation, check whether the answer either (a) contains the abstain string, or (b) references a fact
  from the 20-fact table. If neither — the model is confidently asserting something not in the corpus —
  force-override the response to the abstain string and log the (question, raw model answer) pair for
  review. This is belt-and-suspenders on top of DPO, not a replacement for it.

## 7. Evaluation & Success Metrics

Directly from the PRD, measured via the 36-probe harness at baseline/SFT/DPO:
- Recall accuracy > 90% (20 recall probes)
- Hallucination rate < 5% (8 unanswerable probes; hallucination = 1 − abstain rate)
- No significant drop on general knowledge (8 general probes, compared to baseline)

The baseline → SFT → DPO comparison table is the notebook's existing pattern, reused as-is and relabeled
for EcoBrew.

## 8. Deliverables & Scope (by Thursday, 2026-07-16)

**In scope:**
- Full Jupyter notebook: baseline → SFT (MLX) → parallel SFT (HF/MPS) → DPO (TRL/MPS) → eval →
  serving/export → Gradio demo. Runs end-to-end locally, no internet required after model download.
- `docs/EcoBrew_Product_Facts.md` (done) plus the generated SFT dataset, DPO pairs, and eval set.
- Working local Gradio chat demo with the guardrail layer.

**Explicitly deferred:** presentation slides — may follow on Friday if there is a separate presentation
day.

**Rough time budget:**
- Day 1: data generation (SFT/DPO/eval sets) + baseline/SFT (MLX) working.
- Day 2: HF+PEFT SFT + DPO (TRL/MPS) working — highest-risk day, genuinely new code, not a copy-paste from
  the reference notebook.
- Day 3: eval pass, serving/export path, Gradio + guardrail, end-to-end run-through buffer.

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| TRL `DPOTrainer` hits an unsupported op on `mps` (some PyTorch ops still lag on the MPS backend) | Day 2 is budgeted specifically for this; if truly blocked, CPU fallback for the DPO stage only (slow but correct) is the last resort — still local, just not fast |
| MLX→HF serving-path fuse/convert doesn't round-trip cleanly | Already designed as best-effort with an explicit HF+PEFT/MPS serving fallback (section 5, step 6) |
| Wall-clock training time on M-series is unknown (reference notebook's timings are T4-specific) | Treat the PRD's "<30 min training" NFR as a target, not a gate; reduce `max_steps` if a dry run overshoots |
| 3-day timeline is tight for genuinely new (non-copy-paste) HF/MPS DPO code | Accepted trade-off from keeping DPO as a hard requirement — schedule risk is absorbed by cutting slides (already deferred), not by cutting DPO |
| DPO may not measurably improve abstention on the first attempt (as seen in the reference notebook's own run) | Budget iteration time in Day 2/3 for DPO hyperparameter adjustment (more abstain-category pairs, or raise `beta`), per the reference notebook's own "knobs" guidance |

## 10. Out of Scope (unchanged from PRD)

RAG/retrieval systems, production deployment, large-scale training, presentation slides (deferred to
Friday).
