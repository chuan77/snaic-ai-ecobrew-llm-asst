import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data.generate import build_sft_rows
from scripts.hf_predict import DEVICE
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


if __name__ == "__main__":
    main()
