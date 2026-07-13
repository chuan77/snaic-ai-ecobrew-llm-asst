import torch

from scripts.mlx_predict import SYSTEM_PROMPT

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def hf_predict(question, model, tokenizer, max_new_tokens=48):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(DEVICE)
    output = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True).strip()
