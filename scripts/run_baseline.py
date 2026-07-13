from data.generate import EVAL_QUESTIONS
from evaluation.harness import evaluate
from scripts.mlx_predict import mlx_predict

BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

# HF-compatible (plain transformers/AutoModelForCausalLM) mirror of the same
# Llama-3.2-3B-Instruct family, used by the HF+PEFT/TRL tasks (SFT/DPO on
# torch+mps) since BASE_MODEL above is an MLX-only pre-quantized format that
# transformers cannot load. Ungated fp16 weights published by Unsloth (a
# weights mirror only -- not the Unsloth training framework, which remains
# unused per the plan's Global Constraints).
HF_BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"


def main():
    predict = lambda question: mlx_predict(question, BASE_MODEL)
    scores = evaluate(predict, EVAL_QUESTIONS)
    print(f"[BASE] recall={scores['recall']:.0%} abstain={scores['abstain']:.0%} general={scores['general']:.0%}")
    return scores


if __name__ == "__main__":
    main()
