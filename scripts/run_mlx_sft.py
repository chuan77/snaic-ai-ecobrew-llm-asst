import subprocess
import sys

from data.generate import EVAL_QUESTIONS
from evaluation.harness import evaluate
from scripts.mlx_predict import mlx_predict
from scripts.run_baseline import BASE_MODEL

ADAPTER_PATH = "artifacts/mlx_sft_adapter"


def train():
    subprocess.run(
        [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", BASE_MODEL,
            "--train",
            "--data", "artifacts/mlx_sft_data",
            "--adapter-path", ADAPTER_PATH,
            "--iters", "60",
            "--batch-size", "2",
            "--learning-rate", "2e-4",
        ],
        check=True,
    )


def evaluate_sft():
    predict = lambda question: mlx_predict(question, BASE_MODEL, adapter_path=ADAPTER_PATH)
    scores = evaluate(predict, EVAL_QUESTIONS)
    print(f"[SFT] recall={scores['recall']:.0%} abstain={scores['abstain']:.0%} general={scores['general']:.0%}")
    return scores


if __name__ == "__main__":
    train()
    evaluate_sft()
