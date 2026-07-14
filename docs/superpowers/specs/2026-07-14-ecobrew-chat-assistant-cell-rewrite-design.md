# EcoBrew Interactive Chat Assistant — Cell 11 Rewrite

## Context

`notebooks/EcoBrew_LLM_Customization_Apple_M5_Pro.ipynb` Cell 11 ("EcoBrew Interactive Chat Assistant — Completely Thread-Isolated Load") launches the Gradio chat UI that serves the LoRA-fine-tuned EcoBrew assistant (`models/sft_lora`, base model `mlx-community/Llama-3.2-3B-Instruct-4bit`). The current cell is hard to maintain: it defines a background worker thread + `queue.Queue` pair to work around suspected MLX GPU stream binding issues under Jupyter, then defines the worker loop twice (`mlx_worker_loop`, immediately overridden by `mlx_worker_loop_v2`) with near-duplicate logic. This rewrite replaces it with a single, direct implementation.

## Goal

Re-implement Cell 11 from scratch as a simple, single-pass Gradio chat assistant that:
- Loads the base model + LoRA adapter from `models/sft_lora`.
- Answers EcoBrew product queries (brewing, maintenance, troubleshooting, smart features).
- Preserves the existing guardrail behavior so `TESTs.md` test cases (TC-01..TC-04+) still pass.

## Non-goals

- No standalone script/module extraction — this stays a notebook cell (matches how the rest of the pipeline is authored: cell-by-cell in this notebook).
- No new guardrail logic, no new UI features, no change to the base model or adapter.

## Design

### 1. Model loading

Load directly at the top of the cell, no background thread:

```python
ADAPTER_PATH = str(PROJECT_ROOT / "models" / "sft_lora")

if "model" not in globals() or "tokenizer" not in globals():
    from mlx_lm import load
    model, tokenizer = load(
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        adapter_path=ADAPTER_PATH,
    )
```

Reuses `model`/`tokenizer` if an earlier cell (e.g. Cell 7/8) already loaded them in this kernel session, otherwise loads fresh. This is a re-run optimization, not new behavior — the model+adapter combination is identical either way.

### 2. Chat callback — `ecobrew_chat(message, history)`

One function, no v1/v2 duplication. Same three-step shape as the current Cell 11 body:

1. **Pre-filter** — same `safety_keywords` list (`"python"`, `"write a function"`, `"reverse a string"`, `"ignore"`, `"bypass"`, `"system prompt"`, `"translate"`) → fixed refusal string if matched.
2. **Build messages** — system prompt identical in content to the current cell (role/identity + `ecobrew_knowledge` + hardware limits + the two safety-protocol refusal templates), followed by `history` (Gradio `messages`-format list of `{"role", "content"}` dicts, passed straight through with no tuple-format branch since Gradio 6.20 always supplies dicts), followed by the new user message.
3. **Generate** — `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` → `mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=256, sampler=make_sampler(temp=0.0), verbose=False)`, called inline, synchronously, no queue.
4. **Post-filter** — reject if response contains ` ``` ` or `"def "`.

Exceptions during generation are caught and surfaced as `f"⚠️ Generation error: {e}"`, mirroring today's worker-error message but without the thread boundary.

### 3. Gradio UI

```python
with gr.Blocks(title="EcoBrew Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ☕ EcoBrew Smart Coffee Maker")
    gr.Markdown("### Your Intelligent, Fine-Tuned AI Barista")
    chatbot = gr.Chatbot(height=500, show_label=False, type="messages")
    msg = gr.Textbox(placeholder="Ask anything about brewing profiles, IoT scheduling, or maintenance...", label=None)
    clear = gr.Button("Clear Chat History")

    def respond(message, history):
        response = ecobrew_chat(message, history)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        return "", history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)
```

Same visible UI as today (title, subtitle, chatbot, textbox, clear button). `type="messages"` makes the dict-format history explicit rather than relying on Gradio's implicit default.

### 4. Launch

```python
gr.close_all()
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    prevent_thread_lock=True,
    share=False,
    inbrowser=True,
)
```

Unchanged from today — `gr.close_all()` before `launch()` already guards against orphaned servers on cell re-run; no thread teardown needed since there's no background thread to stop.

## What gets removed

- `queue.Queue` request/response pair.
- `mlx_worker_loop` and `mlx_worker_loop_v2` (duplicate worker loop definitions).
- `threading.Thread` spin-up/teardown and the `mlx_thread.is_alive()` liveness checks.
- The `mxc.stream(mxc.gpu)` / `mxc.clear_cache()` / dummy-array-eval GPU-stream warmup — dropped because generation now runs on the main thread, so there's no cross-thread GPU stream binding to work around. If this turns out to be load-bearing (i.e. testing surfaces a real MLX/Jupyter GPU stream conflict), the fallback is to reintroduce a single worker thread — not two divergent loop definitions.

## Testing

After implementation, run the notebook cell and manually drive a subset of `TESTs.md` against the launched Gradio server at `http://127.0.0.1:7860`:
- TC-01 (bitter brew / domain physics happy path)
- TC-03 (hardware boundary — overkill temperature refusal)
- One prompt-injection case (pre-filter keyword match)

Confirms response content and guardrail refusal strings match expected behavior with the simplified (non-threaded) implementation.

## Addendum (post-implementation): threading is load-bearing after all

Live smoke-testing (driving TC-01/TC-03 through the actual running Gradio HTTP server, not just calling `ecobrew_chat` in-process) reproduced exactly the fallback scenario called out above: `mlx_lm.generate()` failed with `There is no Stream(gpu, 1) in current thread` whenever Gradio's own request-handling thread invoked it, even though the identical call succeeded from the kernel's main thread. Gradio does not run synchronous event callbacks on the thread that loaded the model, and MLX's GPU stream is thread-local — so the direct/non-threaded implementation is not viable as-is.

**Resolution:** reintroduce a single background worker thread that owns the model and serializes all `generate()` calls through a `request_queue`/`response_queue` pair — the same idea as the original cell, but defined once (no `mlx_worker_loop`/`mlx_worker_loop_v2` duplication). The model load moves inside the worker thread's local scope (no more global `model`/`tokenizer` reuse-check, since the worker now owns those variables exclusively). `ecobrew_chat`'s guardrail logic (pre-filter, system prompt, post-filter) is unchanged — it runs on the calling thread and only the `generate()` call is routed through the worker.
