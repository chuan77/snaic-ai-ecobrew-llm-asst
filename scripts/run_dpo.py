import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from data.generate import EVAL_QUESTIONS, build_dpo_pairs
from evaluation.harness import evaluate
from scripts.hf_predict import DEVICE, hf_predict
from scripts.run_baseline import HF_BASE_MODEL
from scripts.run_hf_sft import SFT_ADAPTER_PATH

DPO_ADAPTER_PATH = "artifacts/dpo_adapter"


def _make_prompt(tokenizer, question):
    from scripts.mlx_predict import SYSTEM_PROMPT

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def train():
    tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(HF_BASE_MODEL, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True).to(DEVICE)

    pairs = build_dpo_pairs(eval_questions=EVAL_QUESTIONS)
    dataset = Dataset.from_list(
        [
            {
                "prompt": _make_prompt(tokenizer, pair["prompt"]),
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }
            for pair in pairs
        ]
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            beta=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            max_steps=50,
            learning_rate=5e-6,
            max_length=768,
            logging_steps=10,
            output_dir="artifacts/dpo_out",
            report_to="none",
        ),
    )
    trainer.train()

    # DPOConfig defaults to gradient_checkpointing=True, which leaves the model's
    # gradient-checkpointing hooks active after training. Left active, model.generate()
    # produces degenerate repeated-token output (verified: caused recall/abstain/general
    # to all read 0%) even though use_cache reports True. Disable before eval/inference.
    model.gradient_checkpointing_disable()
    model.eval()

    model.save_pretrained(DPO_ADAPTER_PATH)
    tokenizer.save_pretrained(DPO_ADAPTER_PATH)
    print(f"saved DPO adapter to {DPO_ADAPTER_PATH}")
    return model, tokenizer


def evaluate_dpo(model, tokenizer):
    predict = lambda question: hf_predict(question, model, tokenizer)
    scores = evaluate(predict, EVAL_QUESTIONS)
    print(f"[DPO] recall={scores['recall']:.0%} abstain={scores['abstain']:.0%} general={scores['general']:.0%}")
    return scores


if __name__ == "__main__":
    model, tokenizer = train()
    evaluate_dpo(model, tokenizer)
