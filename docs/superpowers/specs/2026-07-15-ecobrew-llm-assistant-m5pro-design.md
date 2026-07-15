# EcoBrew LLM Assistant — Full Pipeline Notebook (M5 Pro)

## Context

`notebooks/EcoBrew_LLM_Customization_Apple_M5_Pro.ipynb` currently covers setup → SDG (MLX teacher) → curation → SFT (shell-out to `mlx_lm.lora` CLI) → a hardened Gradio assistant. It has no formal eval baseline, no DPO stage, and its knowledge grounding is a short free-text blob rather than discrete facts.

A separate, now-deleted project (`notebooks/ecobrew_closedbook.ipynb` and `scripts/run_baseline.py` / `run_hf_sft.py` / `run_dpo.py` / `evaluation/harness.py` / `data/facts.py`, all removed in the recent restructure — see `git show <sha>:<path>` for the last committed versions) already built a working closed-book pattern: a structured facts list, a recall/abstain/general eval harness, and `peft`/`trl` SFT+DPO on an HF-native model mirror (`unsloth/Llama-3.2-3B-Instruct`). That code is gone from the tree but proven — this design revives its ideas rather than reinventing them.

## Goal

A new notebook, `notebooks/EcoBrew_LLM_Assistant_M5Pro.ipynb`, that takes `Llama-3.2-1B-Instruct-4bit` from a generic base model to a closed-book EcoBrew product assistant, end to end:

0. Project setup
1. Task definition (taxonomy, schema, success criteria, structured facts, project map)
2. Initial evaluation (base model vs. ICL ceiling vs. a genuinely larger reference model)
3. Synthetic data generation & curation, informed by the initial eval
4. SFT via `peft`/`trl` LoRA
5. DPO via `trl`, closing the style/guardrail gap SFT leaves
6. Serving — replacing the old notebook's MLX-based Cells 8/9/11 with a `transformers`/`peft` path, since the DPO adapter is a `peft` artifact, not an MLX one

Every phase gets a markdown cell: `## Phase N: Title` + 2-4 sentences on that phase's objective, matching the existing notebook's heading style but with the "brief description" the current notebook only has for some phases.

## Non-goals

- Not touching the existing `EcoBrew_LLM_Customization_Apple_M5_Pro.ipynb` — it stays as-is; this is a new, separate notebook.
- Not adding OpenAI/Anthropic SDKs or cloud API keys — the "larger model" comparison is a local LM Studio server call.
- Not building a general-purpose eval framework — the harness is sized to this notebook's fact set, not a reusable library (though the logic is lifted near-verbatim from the deleted `evaluation/harness.py`).
- Not attempting full fine-tuning — LoRA only, consistent with the M5 Pro memory/time budget.
- Not resolving how the *old* notebook's Cells 8/9/11 get updated — this is a new notebook; the "interop gap" being closed is between this new notebook's own DPO output and its own serving cells.

## Shared configuration

Extends the current notebook's Cell 0 pattern (root-anchor via `pyproject.toml` marker, `DATA_DIR`/`MODELS_DIR` creation) with:

```python
SDG_MODEL = "mlx-community/gemma-4-e4b-it-4bit"          # unchanged — synthetic-data teacher (MLX)
BASE_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"   # unchanged — the model being evaluated/improved (MLX)
HF_BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"           # HF-native mirror of BASE_MODEL, for peft/trl (no MLX equivalent loadable by transformers)
LMSTUDIO_URL = "http://localhost:1234/api/v1/chat"        # LM Studio's native Responses-style endpoint (verified working, not /v1/chat/completions)
LMSTUDIO_MODEL = "gemma-4-12b-it-mlx"                     # genuinely larger reference model — different weights from SDG_MODEL, so the comparison isn't circular

MODELS_DIR/sft_lora    # peft adapter after Phase 4
MODELS_DIR/dpo_lora    # peft adapter after Phase 5 (trained on top of sft_lora)
```

`DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"` for all `transformers`/`peft` phases (4, 5, 6), matching the deleted `scripts/hf_predict.py`.

## Phase 1: Task Definition

Keeps the current notebook's taxonomy (`Brewing`, `Maintenance`, `Troubleshooting`, `Smart Features`), schema, and success criteria (current Cell 1), plus:

**Structured facts base** (revives `data/facts.py`, trimmed/adapted to this product's existing `ecobrew_knowledge` details rather than the old project's fictitious Verdant/pricing details unless they still fit): ~20-24 entries, each:

```python
{
    "id": int,
    "category": str,        # one of the 4 taxonomy categories, or "company"/"policy"
    "question": str,        # canonical phrasing
    "casual": str,          # casual phrasing variant
    "answer": str,          # grounded answer, on-brand tone
    "accept": [str, ...],   # lowercase keyword(s) that count as a correct hit
}
```

**Project map**: a short markdown cell listing where things live — `data/curated` (train/val), `models/{sft_lora,dpo_lora}` (adapters), the MLX teacher (Phase 3), the LM Studio server (Phase 2, external process) — so a reader can orient without reading every cell.

**Guardrail refusal strings**: the two exact refusal templates (out-of-range-temperature, off-topic) are defined once here as shared constants (`REFUSAL_TEMP`, `REFUSAL_OFFTOPIC`), not inside Phase 6. Phase 2's guardrail probes and Phase 5's DPO pairs both need these before Phase 6's cells exist in notebook execution order, so they're module-level constants referenced by every phase that needs them, defined here and enforced in Phase 6.

## Phase 2: Initial Evaluation

**Eval set**, built from the facts base:
- `recall`: canonical + casual phrasing of each fact → scored by keyword hit-rate against `accept`.
- `unanswerable`: EcoBrew-plausible-but-not-in-facts questions (e.g. "Does the EcoBrew Max have a built-in grinder?") → scored by abstain-phrase match.
- `general`: a few unrelated-knowledge questions (capital of France, etc.) → scored by keyword hit-rate. This checks fine-tuning in Phases 4-5 doesn't destroy general capability; it is independent of the *product* off-topic-refusal guardrail enforced later in Phase 6's `ecobrew_assistant` wrapper.
- 2 guardrail probes (out-of-range temperature request, off-topic/prompt-injection request) — scored by exact match against the shared `REFUSAL_TEMP`/`REFUSAL_OFFTOPIC` constants (Phase 1), so probes and Phase 6's enforcement can't drift apart.

Scoring logic revives the deleted `evaluation/harness.py` almost verbatim (`_norm`, `_is_abstain`, `hit_rate`, `evaluate`), parameterized over a `predict_fn` so the same harness scores all three backends below.

**Three-way comparison**, all against the same eval set:
1. `BASE_MODEL` (MLX), zero-shot — the model this whole notebook exists to improve.
2. `BASE_MODEL` (MLX), few-shot ICL — a handful of facts inserted as in-context exemplars in the system/user prompt, establishing the prompting-only ceiling before any training.
3. `LMSTUDIO_MODEL` via `LMSTUDIO_URL`:
   ```python
   resp = requests.post(LMSTUDIO_URL, json={
       "model": LMSTUDIO_MODEL,
       "system_prompt": SYSTEM_PROMPT,
       "input": question,
   }, timeout=60)
   answer = resp.json()["output"][0]["content"]
   ```
   Wrapped in `try/except requests.RequestException`, printing a warning and substituting `None`/skip if the LM Studio server isn't running — the notebook must not hard-fail on this cell.

Results collected into one `pandas` comparison table (rows = eval question types, columns = the three backends), and category-level breakdown used to weight Phase 3.

## Phase 3: Synthetic Data Generation & Curation

Two generation modes feeding one curated set:
- **Fact-phrasing expansion**: for each fact, template a handful of phrasing variants (reuses the deleted `data/generate.py`'s `_variants` idea) — gives the model many surface forms of the same grounded answer, which is what closed-book recall training needs.
- **Teacher-elaborated generation** (current notebook's Cell 5 logic, unchanged): Gemma-4-e4b (MLX) generates open-ended troubleshooting/maintenance-style Q&A grounded in `ecobrew_knowledge`, for natural phrasing diversity beyond the fixed facts.

Sample allocation across the 4 taxonomy categories is weighted toward whichever categories scored worst in Phase 2's comparison table (a simple proportional weighting, not a new algorithm).

Curation (current notebook's Cell 6 logic, unchanged): length filter, train/val split, `{"messages": [...]}` chat format written to `data/curated/{train,valid}.jsonl` — already directly loadable by `datasets.Dataset.from_json` for Phase 4, no separate HF-specific format needed.

## Phase 4: Supervised Fine-Tuning (SFT)

A short markdown explainer on LoRA (what `r`/`lora_alpha`/`target_modules` control, why LoRA fits the M5 Pro memory budget vs. full fine-tuning) precedes the code.

```python
tokenizer = AutoTokenizer.from_pretrained(HF_BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(HF_BASE_MODEL, torch_dtype=torch.bfloat16).to(DEVICE)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
))

train_ds = Dataset.from_json(str(CURATED_DIR / "train.jsonl"))
val_ds = Dataset.from_json(str(CURATED_DIR / "valid.jsonl"))

trainer = SFTTrainer(
    model=model, processing_class=tokenizer,
    train_dataset=train_ds, eval_dataset=val_ds,
    args=SFTConfig(max_length=1024, per_device_train_batch_size=2, gradient_accumulation_steps=4,
                   max_steps=60, learning_rate=2e-4, logging_steps=10,
                   output_dir=str(MODELS_DIR / "sft_out"), report_to="none"),
)
trainer.train()
model.gradient_checkpointing_disable()  # required: left enabled, generate() degenerates post-training (learned the hard way in the deleted run_hf_sft.py — carrying the fix forward, not rediscovering it)
model.eval()
model.save_pretrained(str(MODELS_DIR / "sft_lora"))
tokenizer.save_pretrained(str(MODELS_DIR / "sft_lora"))
```

`trl.SFTTrainer`/`SFTConfig` is used instead of a hand-rolled `Trainer`+`DataCollatorForLanguageModeling` — this exact model/task combination already worked this way in the deleted `scripts/run_hf_sft.py`, including the gradient-checkpointing gotcha above; re-deriving it with a raw `Trainer` risks silently reintroducing that bug.

Evaluate via Phase 2's harness against the same eval set (now via `hf_predict`-style generation on `mps`). A markdown decision cell: if recall is weak, bump `max_steps` and re-run rather than redesigning — "train a short adapter, decide whether to extend" from the brief.

## Phase 5: Direct Preference Optimization (DPO)

Chosen/rejected pairs, all built from data already in hand (no new generation):

- **Recall-confidence pairs**: known fact → chosen = correct answer, rejected = unnecessary abstain.
- **Anti-confusion pairs**: known fact → chosen = correct answer, rejected = a *different* fact's answer.
- **Anti-fabrication pairs**: not-in-facts prompt → chosen = abstain, rejected = a fabricated-sounding answer.
- **Guardrail pairs**: out-of-range-temperature / off-topic prompt → chosen = `REFUSAL_TEMP`/`REFUSAL_OFFTOPIC` (Phase 1 constants), rejected = a compliant-but-policy-violating answer.
- **Quality-contrast pairs**: on-topic question → chosen = curated synthetic (Phase 3) answer, rejected = `BASE_MODEL`'s own zero-shot answer captured during Phase 2 — reusing Phase 2's baseline output as the negative signal instead of generating anything new.

```python
base_model = AutoModelForCausalLM.from_pretrained(HF_BASE_MODEL, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, str(MODELS_DIR / "sft_lora"), is_trainable=True).to(DEVICE)

trainer = DPOTrainer(
    model=model, ref_model=None, processing_class=tokenizer,
    train_dataset=dpo_dataset,  # {"prompt", "chosen", "rejected"}
    args=DPOConfig(beta=0.1, per_device_train_batch_size=1, gradient_accumulation_steps=4,
                   max_steps=50, learning_rate=5e-6, max_length=768, logging_steps=10,
                   output_dir=str(MODELS_DIR / "dpo_out"), report_to="none"),
)
trainer.train()
model.gradient_checkpointing_disable()  # same gotcha as Phase 4 — DPOConfig also defaults this on
model.eval()
model.save_pretrained(str(MODELS_DIR / "dpo_lora"))
```

Evaluated on the same harness, plus an explicit pass/fail check on the two guardrail probes (exact-string match), since that's the behavior DPO exists to lock in.

## Phase 6: Serving (interop-gap fix)

Replaces the old notebook's MLX-based Cells 8/9/11 with a `transformers`/`peft` equivalent, since `models/dpo_lora` is a `peft` artifact `mlx_lm.load()` cannot read:

- **Post-training test cell**: load `HF_BASE_MODEL` + `PeftModel.from_pretrained(..., str(MODELS_DIR / "dpo_lora"))` on `mps`, run the notebook's `test_queries` through it.
- **`ecobrew_assistant(query)`**: identical logic to the current notebook's Cell 9 (keyword pre-filter → system prompt with hardware limits + exact refusal templates → generate → code-leak post-filter) — only the generation call changes, from `mlx_lm.generate` to `model.generate()` + `tokenizer.apply_chat_template(..., return_tensors="pt").to(DEVICE)`.
- **Regression suite**: current notebook's Cell 10, reused unchanged — it only calls `ecobrew_assistant()`, already backend-agnostic.
- **Gradio chat cell**: same single-worker-thread + `queue.Queue` architecture as the current notebook's Cell 11 (necessary because Gradio's request thread doesn't share GPU/device context with the loading thread — this was a real, hard-won fix in the current notebook's history, not incidental), same `_flatten_history_content` history-flattening fix, but the worker loads/generates via `transformers`/`peft` instead of `mlx_lm`.

## Testing / Verification

No unit-test suite (this is a notebook, matching the existing project's testing style) — verification is running each phase's cell and checking:
- Phase 2: comparison table populates for all 3 backends (or LM Studio row is clearly marked skipped, not silently empty).
- Phase 4/5: adapter directories (`models/sft_lora`, `models/dpo_lora`) contain `adapter_model.safetensors` + `adapter_config.json` after training; harness scores improve recall vs. Phase 2's zero-shot baseline.
- Phase 6: regression suite (current notebook's Cell 10 logic) reports Pass on the guardrail test cases, driven against the live Gradio server the same way `TESTs.md` currently verifies the existing notebook.

## Known gotchas carried forward (not to rediscover)

1. `gradient_checkpointing` defaults on in both `SFTConfig` and `DPOConfig` — must be disabled before eval/inference or `generate()` produces degenerate repeated-token output despite `use_cache` reporting `True`.
2. MLX's GPU stream is thread-local — Gradio's request-handling thread cannot call a model loaded on another thread; the single-worker-thread/queue pattern is required, not optional, wherever MLX generation runs under Gradio. `torch`/`mps` in Phase 6 does not share this specific constraint (PyTorch device context isn't thread-bound the same way), so the worker-thread there is a deliberate choice for architectural consistency with the rest of the notebook, not a technical requirement — worth knowing so it isn't mistaken for load-bearing if it's ever simplified away.
3. Llama tokenizers ship with no `pad_token` — set `tokenizer.pad_token = tokenizer.eos_token` before batching in Phase 4/5.
4. LM Studio's native endpoint is `/api/v1/chat` (`system_prompt`/`input` → `output[0].content`), not the OpenAI-compatible `/v1/chat/completions` — verified directly against the running server; the two return different response shapes.
