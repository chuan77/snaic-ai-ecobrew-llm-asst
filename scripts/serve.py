import subprocess
import sys
from pathlib import Path

from scripts.run_baseline import BASE_MODEL, HF_BASE_MODEL
from scripts.run_dpo import DPO_ADAPTER_PATH

MLX_FUSED_PATH = "artifacts/mlx_fused_model"

# Sanity-check questions used to verify the fused MLX model actually produces
# sensible output rather than silently-corrupted garbage. `mlx_lm.fuse` can
# "succeed" (exit 0, write output files) even when the adapter's LoRA deltas
# don't match the target base model's weights -- this happens here because
# DPO_ADAPTER_PATH was trained against HF_BASE_MODEL ("unsloth/Llama-3.2-3B-Instruct",
# a plain-HF fp16 mirror) while BASE_MODEL ("mlx-community/Llama-3.2-3B-Instruct-4bit")
# is a separately-quantized MLX checkpoint of the same nominal family. Fusing
# across those two independently-produced weight sets is not guaranteed to be
# numerically meaningful even when no exception is raised.
_SANITY_QUESTION = "What does the EcoBrew One cost?"
_SANITY_EXPECTED_SUBSTRING = "$89"


def _looks_sane(text):
    """Reject degenerate/garbage output from a "successful" fuse.

    Checks for the known failure modes of a corrupted fuse: empty output,
    heavy repetition of a single token/short pattern, non-ASCII noise, or a
    reply that doesn't even contain the one fact the fused model is being
    quizzed on.
    """
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    # Degenerate repetition check: split into words, see if one word dominates.
    words = stripped.split()
    if len(words) >= 4:
        most_common_count = max(words.count(w) for w in set(words))
        if most_common_count / len(words) > 0.6:
            return False
    # Mostly-non-printable / non-ASCII noise check.
    printable_ratio = sum(1 for c in stripped if c.isprintable()) / len(stripped)
    if printable_ratio < 0.9:
        return False
    # The sanity question has a known factual answer; a genuinely fused model
    # should reproduce it verbatim (it's a memorized training fact, not a
    # generalization test), so its absence is a strong corruption signal.
    if _SANITY_EXPECTED_SUBSTRING not in stripped:
        return False
    return True


def try_build_mlx_serving_model():
    """Best-effort: fuse the DPO adapter and convert to MLX for fast local inference.

    Returns True on success, False if the conversion should be skipped in favor
    of the HF+PEFT/MPS fallback (per design doc section 5, step 6).
    """
    try:
        subprocess.run(
            [
                sys.executable, "-m", "mlx_lm.fuse",
                "--model", BASE_MODEL,
                "--adapter-path", DPO_ADAPTER_PATH,
                "--save-path", MLX_FUSED_PATH,
                "--dequantize",
            ],
            check=True,
        )
        if not Path(MLX_FUSED_PATH).exists():
            return False

        # The subprocess call above can "succeed" (exit 0) even when the adapter
        # was trained against a different base model's weights, since fusing
        # mismatched LoRA deltas onto unrelated weights is not itself an error --
        # it just produces numerical garbage. Verify with a real generation
        # before trusting this path.
        from scripts.mlx_predict import mlx_predict

        try:
            sample = mlx_predict(_SANITY_QUESTION, MLX_FUSED_PATH)
        except Exception:
            return False

        return _looks_sane(sample)
    except subprocess.CalledProcessError:
        return False


def get_predict_fn():
    if try_build_mlx_serving_model():
        from scripts.mlx_predict import mlx_predict

        return lambda question: mlx_predict(question, MLX_FUSED_PATH)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.hf_predict import DEVICE, hf_predict

    tokenizer = AutoTokenizer.from_pretrained(DPO_ADAPTER_PATH)
    # NOTE: base model for the fallback must be HF_BASE_MODEL, not BASE_MODEL.
    # BASE_MODEL ("mlx-community/Llama-3.2-3B-Instruct-4bit") is an MLX-only
    # pre-quantized checkpoint that transformers.AutoModelForCausalLM cannot
    # load at all, and it is not the model DPO_ADAPTER_PATH's LoRA weights were
    # trained against anyway (see artifacts/dpo_adapter/adapter_config.json's
    # base_model_name_or_path == HF_BASE_MODEL). Loading BASE_MODEL here would
    # either crash or silently produce a mismatched, garbage-output model.
    base_model = AutoModelForCausalLM.from_pretrained(HF_BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, DPO_ADAPTER_PATH).to(DEVICE)
    model.eval()
    return lambda question: hf_predict(question, model, tokenizer)
