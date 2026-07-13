from mlx_lm import generate, load

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question in one short sentence. "
    "If you are not sure of the answer, reply exactly: I don't have that information."
)

_cache = {}


def _load_cached(model_path, adapter_path):
    key = (model_path, adapter_path)
    if key not in _cache:
        _cache[key] = load(model_path, adapter_path=adapter_path)
    return _cache[key]


def mlx_predict(question, model_path, adapter_path=None, max_tokens=48):
    model, tokenizer = _load_cached(model_path, adapter_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False).strip()
