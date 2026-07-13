# Architecture Decision Records (ADRs)

## ADR-001: Fine-Tuning Framework
**Status:** Accepted  
**Date:** 2026-07-13

### Context
Need efficient LoRA fine-tuning on Apple Silicon MacBook.

### Decision
Use **Apple MLX** framework as primary, with Hugging Face PEFT as fallback.

### Rationale
- MLX is optimized for Apple Silicon (unified memory, fast).
- Faster training than standard HF on Mac.
- Good community examples for LoRA.

### Alternatives Considered
- Pure Hugging Face Transformers + PEFT (slower on Mac).

## ADR-002: User Interface
**Status:** Accepted  
**Date:** 2026-07-13

### Context
Need simple chat demo for presentation.

### Decision
Use **Gradio** `ChatInterface`.

### Rationale
- Minimal code.
- Excellent streaming support.
- Runs locally in browser.
- Easy to style.

### Alternatives
- Streamlit (heavier for simple chat).

## ADR-003: Base Model
**Status:** Accepted  
**Date:** 2026-07-13

### Decision
`mlx-community/Mistral-7B-Instruct-4bit` or Phi-3-mini.

### Rationale
Balance of capability, speed, and memory usage on MacBook.