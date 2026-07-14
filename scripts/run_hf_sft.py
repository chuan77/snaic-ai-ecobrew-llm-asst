import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data.generate import EVAL_QUESTIONS, build_sft_rows
from evaluation.harness import evaluate
from scripts.hf_predict import DEVICE, hf_predict
from scripts.mlx_predict import SYSTEM_PROMPT
from scripts.run_baseline import HF_BASE_MODEL

SFT_ADAPTER_PATH = "artifacts/hf_sft_adapter"


def _sft_text(tokenizer, row):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    tokenizer = AutoTokenizer.from_pretrained(HF_BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(HF_BASE_MODEL, torch_dtype=torch.float16).to(DEVICE)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )

    rows = build_sft_rows()
    dataset = Dataset.from_dict({"text": [_sft_text(tokenizer, row) for row in rows]})

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=1024,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            logging_steps=10,
            output_dir="artifacts/hf_sft_out",
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(SFT_ADAPTER_PATH)
    tokenizer.save_pretrained(SFT_ADAPTER_PATH)
    print(f"saved HF SFT adapter to {SFT_ADAPTER_PATH}")

    # SFTConfig defaults to gradient_checkpointing=True, which leaves the model's
    # gradient-checkpointing hooks active after training (same issue documented in
    # run_dpo.py) -- disable before eval/inference or generate() produces garbage.
    model.gradient_checkpointing_disable()
    model.eval()
    return model, tokenizer


def evaluate_sft(model, tokenizer):
    predict = lambda question: hf_predict(question, model, tokenizer)
    scores = evaluate(predict, EVAL_QUESTIONS)
    print(f"[HF-SFT] recall={scores['recall']:.0%} abstain={scores['abstain']:.0%} general={scores['general']:.0%}")
    return scores


if __name__ == "__main__":
    model, tokenizer = main()
    evaluate_sft(model, tokenizer)
