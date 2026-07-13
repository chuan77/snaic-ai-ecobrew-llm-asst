1. Project Title
EcoBrew Closed-Book Product Assistant – Knowledge Injection via SFT + DPO on Local MacBook

2. Objective
Build a closed-book AI assistant that answers questions about the EcoBrew Smart Coffee Maker from memory only (no retrieval or web lookup).
Use SFT to inject ~20 product facts, then DPO to teach the model when to abstain politely instead of hallucinating.
3. Problem & Gap
Base models have never seen EcoBrew → they either refuse or invent facts.
We close this gap by injecting facts directly into the model weights.
4. Chosen Technique

Stage 1: SFT – Teach the facts (recall ↑).
Stage 2: DPO – Teach honest abstention (hallucination ↓) using preference pairs built from the corpus.
Guardrails – Extra output validation + logging for unknown questions.

5. Scope

~20 detailed product facts (specs, features, policies, troubleshooting, warranty).
Dataset: ~100 SFT examples + DPO preference pairs.
Model: Small efficient model (Mistral-7B-4bit or Phi-3 via Unsloth/MLX).
Evaluation: 3 metrics – Recall of taught facts, Abstention on unknowns, No drop on general knowledge.
Deliverables: Full Jupyter Notebook (baseline → SFT → DPO → Gradio demo), PRD, ADRs, code, data.

6. Local MacBook Setup

Run entirely in VS Code.
Use MLX / Unsloth for Apple Silicon efficiency.
Simple Gradio chat UI with guardrails.

7. Timeline (by Friday)

Data + Notebook setup → Today/Tomorrow
SFT + DPO training + evaluation → Mid-week
Gradio demo + slides → Final day
Total build time: Fits ~2.5 hours core work + testing.