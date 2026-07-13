import json
from pathlib import Path

from data.generate import build_sft_rows
from scripts.mlx_predict import SYSTEM_PROMPT


def main(out_dir="artifacts/mlx_sft_data"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = build_sft_rows()

    lines = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        lines.append(json.dumps({"messages": messages}))

    (out_path / "train.jsonl").write_text("\n".join(lines) + "\n")
    (out_path / "valid.jsonl").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} rows to {out_path}")


if __name__ == "__main__":
    main()
