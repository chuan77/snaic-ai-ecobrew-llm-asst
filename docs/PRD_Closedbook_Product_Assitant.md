# Product Requirements Document (PRD)
## Closed-Book EcoBrew Smart Coffee Maker Assistant

**Version:** 1.0  
**Date:** July 13, 2026  
**Project Type:** Mini-project for LLM Fine-Tuning Course  
**Status:** Draft

### 1. Overview
A local, closed-book AI assistant that answers questions about the EcoBrew Smart Coffee Maker using only injected knowledge (no retrieval or external lookup). Built with SFT/LoRA on Apple Silicon and includes a simple Gradio chat interface.

### 2. Objectives
- Demonstrate closed-book recall of product facts.
- Prevent hallucinations on product queries.
- Maintain general knowledge.
- Provide an interactive local demo.

### 3. Target Users
- Project evaluators / instructors
- Team members (for presentation)

### 4. Key Features
1. Closed-book QA on ~20 product facts.
2. Graceful abstention on out-of-scope questions.
3. Local Gradio chat UI with streaming responses.
4. Evaluation scripts for 3 metrics.

### 5. Non-Functional Requirements
- Runs entirely on local MacBook (M-series).
- Training < 30 min.
- Inference responsive (< 2s first token).
- No internet required after setup.

### 6. Product Facts (Core Knowledge)
See [EcoBrew_Product_Facts.md](EcoBrew_Product_Facts.md) for the full table of 20 invented facts (company, pricing, specs, features, policies, warranty, troubleshooting).

### 7. Success Metrics
- Recall accuracy > 90% on taught facts.
- Hallucination rate < 5% on unknowns.
- No significant drop on general benchmarks.

### 8. Out of Scope
- RAG / retrieval systems.
- Production deployment.
- Large-scale training.