from data.generate import EVAL_QUESTIONS
from evaluation.harness import evaluate
from scripts.mlx_predict import mlx_predict

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"


def main():
    predict = lambda question: mlx_predict(question, BASE_MODEL)
    scores = evaluate(predict, EVAL_QUESTIONS)
    print(f"[BASE] recall={scores['recall']:.0%} abstain={scores['abstain']:.0%} general={scores['general']:.0%}")
    return scores


if __name__ == "__main__":
    main()
