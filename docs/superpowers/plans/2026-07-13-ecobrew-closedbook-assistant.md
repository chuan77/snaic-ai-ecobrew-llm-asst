# EcoBrew Closed-Book Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a closed-book EcoBrew Smart Coffee Maker assistant — SFT (MLX) to inject 20 facts, DPO (HF+PEFT+TRL on MPS) to teach graceful abstention, evaluated on recall/hallucination/general-knowledge, served through a local Gradio demo with an output-validation guardrail.

**Architecture:** A single source-of-truth fact table drives generated SFT rows, DPO preference pairs, and an eval harness (all pure-Python, unit-tested). Training is a hybrid: `mlx-lm` LoRA for the ADR-mandated MLX SFT stage, and an independent HF+PEFT LoRA SFT run on `mps` that feeds TRL's `DPOTrainer` for DPO (Unsloth is CUDA-only and unusable on Mac, so this is a from-scratch adaptation, not a port). A shared `evaluate()` function scores every stage. Serving prefers a fused MLX model for speed, falling back to HF+PEFT on `mps` if the MLX conversion doesn't round-trip cleanly.

**Tech Stack:** Python 3.11, `mlx` + `mlx-lm`, `torch` (MPS backend) + `transformers` + `peft` + `trl`, `gradio`, `pytest`.

## Global Constraints

- Base model: `microsoft/Phi-3-mini-4k-instruct`, 4-bit quantized for the MLX path (per design doc section 3).
- Platform: local Apple Silicon Mac only — no Colab, no internet required after model download (PRD NFR).
- DPO is a hard requirement — no scope fallback to SFT-only.
- Do not use Unsloth anywhere — it is CUDA-only and will not run on this machine.
- System prompt (verbatim, used everywhere a model is prompted): `"You are a helpful assistant. Answer the question in one short sentence. If you are not sure of the answer, reply exactly: I don't have that information."`
- Abstain string (verbatim, used everywhere): `"I don't have that information."`
- Success thresholds (PRD): recall > 90%, hallucination < 5% (i.e. abstain > 95% on unanswerable probes), no significant drop on general-knowledge vs. baseline.
- Deadline: working notebook + Gradio demo by Thursday 2026-07-16. Slides are out of scope for this plan.

---

### Task 1: Project scaffolding and the facts table

**Files:**
- Create: `requirements.txt`
- Create: `data/__init__.py`
- Create: `data/facts.py`
- Test: `tests/test_facts.py`

**Interfaces:**
- Produces: `data.facts.FACTS` — `list[dict]`, each dict has keys `id: int`, `category: str`, `question: str`, `answer: str`, `accept: list[str]`. 20 entries.

- [ ] **Step 1: Create `requirements.txt`**

```text
mlx
mlx-lm
torch
transformers
peft
trl
accelerate
gradio
pytest
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: all packages install without error (this will take a few minutes for `torch`).

- [ ] **Step 3: Write the failing test for the facts table**

Create `tests/test_facts.py`:

```python
from data.facts import FACTS


def test_facts_has_20_entries():
    assert len(FACTS) == 20


def test_facts_have_required_keys():
    required = {"id", "category", "question", "answer", "accept"}
    for fact in FACTS:
        assert required <= set(fact.keys())


def test_facts_ids_are_unique_and_sequential():
    ids = [fact["id"] for fact in FACTS]
    assert ids == list(range(1, 21))


def test_facts_accept_lists_are_lowercase():
    for fact in FACTS:
        for accept in fact["accept"]:
            assert accept == accept.lower()
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `pytest tests/test_facts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.facts'` (or similar import error).

- [ ] **Step 5: Create `data/__init__.py`**

Empty file.

- [ ] **Step 6: Write `data/facts.py`**

```python
FACTS = [
    {"id": 1, "category": "company", "question": "Where is EcoBrew's parent company headquartered?",
     "answer": "EcoBrew is made by Verdant Home Appliances, headquartered in Portland, Oregon.",
     "accept": ["portland"]},
    {"id": 2, "category": "company", "question": "When was Verdant Home Appliances founded?",
     "answer": "Verdant Home Appliances was founded in 2020.",
     "accept": ["2020"]},
    {"id": 3, "category": "company", "question": "Who founded and leads Verdant Home Appliances?",
     "answer": "Verdant Home Appliances was founded and is led by Maria Chen.",
     "accept": ["maria chen", "chen"]},
    {"id": 4, "category": "product", "question": "What is EcoBrew's flagship product line?",
     "answer": "EcoBrew's flagship product line is the EcoBrew Smart Coffee Maker.",
     "accept": ["smart coffee maker"]},
    {"id": 5, "category": "pricing", "question": "What does the EcoBrew One cost?",
     "answer": "The EcoBrew One costs $89.",
     "accept": ["89"]},
    {"id": 6, "category": "pricing", "question": "What does the EcoBrew Pro cost?",
     "answer": "The EcoBrew Pro costs $149.",
     "accept": ["149"]},
    {"id": 7, "category": "pricing", "question": "What does the EcoBrew Max cost?",
     "answer": "The EcoBrew Max costs $219.",
     "accept": ["219"]},
    {"id": 8, "category": "specs", "question": "How many cups per pot does the EcoBrew Pro brew?",
     "answer": "The EcoBrew Pro brews up to 12 cups per pot.",
     "accept": ["12"]},
    {"id": 9, "category": "feature", "question": "What is EcoBrew's companion app called?",
     "answer": "EcoBrew's companion app is called GreenCup.",
     "accept": ["greencup"]},
    {"id": 10, "category": "connectivity", "question": "What Wi-Fi standard do the EcoBrew Pro and Max support?",
     "answer": "The EcoBrew Pro and Max support Wi-Fi 6.",
     "accept": ["wi-fi 6", "wifi 6", "wi fi 6"]},
    {"id": 11, "category": "feature", "question": "What kind of filter does EcoBrew use?",
     "answer": "EcoBrew uses a reusable stainless-steel mesh filter, no paper filters needed.",
     "accept": ["stainless-steel", "stainless steel", "mesh filter"]},
    {"id": 12, "category": "warranty", "question": "What warranty comes with an EcoBrew coffee maker?",
     "answer": "EcoBrew coffee makers come with a 2-year standard warranty.",
     "accept": ["2-year", "2 year", "two year", "two-year"]},
    {"id": 13, "category": "policy", "question": "What is EcoBrew's return policy window?",
     "answer": "EcoBrew allows returns within a 45-day window.",
     "accept": ["45"]},
    {"id": 14, "category": "policy", "question": "What are EcoBrew's customer support hours?",
     "answer": "EcoBrew customer support runs 09:00 to 18:00 EST, Monday to Friday.",
     "accept": ["09:00", "9:00", "18:00"]},
    {"id": 15, "category": "troubleshooting", "question": "How often should an EcoBrew be descaled?",
     "answer": "EcoBrew should be descaled every 3 months using a citric-acid descaling solution.",
     "accept": ["3 months", "three months"]},
    {"id": 16, "category": "specs", "question": "How many watts does the EcoBrew Max draw?",
     "answer": "The EcoBrew Max draws 800 watts; Eco Mode cuts power draw by 30%.",
     "accept": ["800"]},
    {"id": 17, "category": "sustainability", "question": "What is the EcoBrew housing made from?",
     "answer": "The EcoBrew housing is made from 70% recycled ocean-bound plastic.",
     "accept": ["recycled ocean-bound plastic", "70%"]},
    {"id": 18, "category": "subscription", "question": "How much does the EcoBrew+ subscription cost?",
     "answer": "The EcoBrew+ subscription costs $4.99 per month and includes recipe presets and auto filter reorder.",
     "accept": ["4.99"]},
    {"id": 19, "category": "feature", "question": "What is the EcoBrew Max's built-in voice assistant called?",
     "answer": "The EcoBrew Max's built-in voice assistant is called Sprout.",
     "accept": ["sprout"]},
    {"id": 20, "category": "feature", "question": "After how long does EcoBrew auto-shut off?",
     "answer": "EcoBrew auto-shuts off after 40 minutes of inactivity.",
     "accept": ["40 minutes", "40 min"]},
]
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `pytest tests/test_facts.py -v`
Expected: 4 passed.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt data/__init__.py data/facts.py tests/test_facts.py
git commit -m "feat: add EcoBrew facts table and project scaffolding"
```

---

### Task 2: SFT row and eval-question generation

**Files:**
- Create: `data/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `data.facts.FACTS` (Task 1).
- Produces:
  - `data.generate.ABSTAIN: str`
  - `data.generate.build_sft_rows(facts=FACTS) -> list[dict]` — each dict `{"question": str, "answer": str}`.
  - `data.generate.EVAL_QUESTIONS: list[dict]` — each dict `{"id": str, "type": "recall"|"unanswerable"|"general", "question": str, "accept": list[str]}`, 36 entries (20 recall + 8 unanswerable + 8 general).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate.py`:

```python
from data.facts import FACTS
from data.generate import ABSTAIN, EVAL_QUESTIONS, build_sft_rows


def test_sft_rows_count_and_shape():
    rows = build_sft_rows()
    assert len(rows) == 120
    assert all(set(row.keys()) == {"question", "answer"} for row in rows)


def test_sft_rows_cover_every_fact_answer():
    rows = build_sft_rows()
    row_answers = {row["answer"] for row in rows}
    fact_answers = {fact["answer"] for fact in FACTS}
    assert row_answers == fact_answers


def test_sft_rows_have_six_variants_per_fact():
    rows = build_sft_rows()
    for fact in FACTS:
        matching = [row for row in rows if row["answer"] == fact["answer"]]
        assert len(matching) == 6


def test_eval_questions_shape():
    assert len(EVAL_QUESTIONS) == 36
    types = [q["type"] for q in EVAL_QUESTIONS]
    assert types.count("recall") == 20
    assert types.count("unanswerable") == 8
    assert types.count("general") == 8


def test_eval_recall_questions_differ_from_sft_phrasing():
    rows = build_sft_rows()
    sft_questions = {row["question"] for row in rows}
    recall_questions = {q["question"] for q in EVAL_QUESTIONS if q["type"] == "recall"}
    assert sft_questions.isdisjoint(recall_questions)


def test_abstain_string_is_exact():
    assert ABSTAIN == "I don't have that information."
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.generate'`.

- [ ] **Step 3: Write `data/generate.py`**

```python
from data.facts import FACTS

ABSTAIN = "I don't have that information."


def _variants(question):
    base = question.rstrip("?").strip()
    lower_first = base[0].lower() + base[1:]
    return [
        f"{base}?",
        f"{base}, please?",
        f"Quick question: {lower_first}?",
        f"Quick question: {lower_first}, please?",
        f"Could you tell me {lower_first}?",
        f"I'd like to know {lower_first}.",
    ]


def build_sft_rows(facts=FACTS):
    rows = []
    for fact in facts:
        for question in _variants(fact["question"]):
            rows.append({"question": question, "answer": fact["answer"]})
    return rows


_RECALL_PROBES = [
    ("Which city is EcoBrew's maker based in?", ["portland"]),
    ("In what year did Verdant Home Appliances get started?", ["2020"]),
    ("Who is the founder of Verdant Home Appliances?", ["maria chen", "chen"]),
    ("What is the name of EcoBrew's main product line?", ["smart coffee maker"]),
    ("How much is the entry-level EcoBrew One?", ["89"]),
    ("What's the price tag on an EcoBrew Pro?", ["149"]),
    ("How much does the top-tier EcoBrew Max sell for?", ["219"]),
    ("What's the brewing capacity of the EcoBrew Pro, in cups?", ["12"]),
    ("What is the name of the app that pairs with EcoBrew?", ["greencup"]),
    ("Which Wi-Fi generation is supported by the Pro and Max models?", ["wi-fi 6", "wifi 6", "wi fi 6"]),
    ("Does EcoBrew use paper filters or something else?", ["stainless-steel", "stainless steel", "mesh filter"]),
    ("How long is the standard warranty on an EcoBrew?", ["2-year", "2 year", "two year", "two-year"]),
    ("Within how many days can you return an EcoBrew?", ["45"]),
    ("During what hours can you reach EcoBrew's support team?", ["09:00", "9:00", "18:00"]),
    ("How frequently does the EcoBrew need descaling?", ["3 months", "three months"]),
    ("What is the power draw of the EcoBrew Max in watts?", ["800"]),
    ("What material is used for the EcoBrew's housing?", ["recycled ocean-bound plastic", "70%"]),
    ("What is the monthly price of EcoBrew+?", ["4.99"]),
    ("What's the name of the voice assistant built into the EcoBrew Max?", ["sprout"]),
    ("After what period of inactivity does EcoBrew turn itself off?", ["40 minutes", "40 min"]),
]

_UNANSWERABLE_PROBES = [
    "Does the EcoBrew Max have a built-in coffee bean grinder?",
    "What color options are available for the EcoBrew Pro?",
    "Can the EcoBrew connect to Amazon Alexa?",
    "What is Verdant Home Appliances' annual revenue?",
    "Does EcoBrew offer a student discount?",
    "What is the weight of the EcoBrew Max?",
    "Is there an EcoBrew Mini model?",
    "Does the EcoBrew Pro support a cold brew mode?",
]

_GENERAL_PROBES = [
    ("What is the capital of France?", ["paris"]),
    ("What is 2 + 2?", ["4", "four"]),
    ("What is the chemical symbol for water?", ["h2o"]),
    ("Who wrote Romeo and Juliet?", ["shakespeare"]),
    ("What is the largest planet in our solar system?", ["jupiter"]),
    ("How many continents are there on Earth?", ["7", "seven"]),
    ("What color results from mixing blue and yellow paint?", ["green"]),
    ("What is the boiling point of water in degrees Celsius?", ["100"]),
]


def _build_eval_questions():
    questions = []
    for i, (question, accept) in enumerate(_RECALL_PROBES, start=1):
        questions.append({"id": f"r{i:02d}", "type": "recall", "question": question, "accept": accept})
    for i, question in enumerate(_UNANSWERABLE_PROBES, start=1):
        questions.append({"id": f"u{i:02d}", "type": "unanswerable", "question": question, "accept": []})
    for i, (question, accept) in enumerate(_GENERAL_PROBES, start=1):
        questions.append({"id": f"g{i:02d}", "type": "general", "question": question, "accept": accept})
    return questions


EVAL_QUESTIONS = _build_eval_questions()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_generate.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add data/generate.py tests/test_generate.py
git commit -m "feat: generate SFT rows and eval question set from the facts table"
```

---

### Task 3: DPO pair generation

**Files:**
- Modify: `data/generate.py`
- Modify: `tests/test_generate.py`

**Interfaces:**
- Consumes: `data.facts.FACTS`, `data.generate.ABSTAIN`, `data.generate.EVAL_QUESTIONS` (Tasks 1-2).
- Produces: `data.generate.build_dpo_pairs(facts=FACTS, eval_questions=EVAL_QUESTIONS, seed=3407) -> list[dict]` — each dict `{"prompt": str, "chosen": str, "rejected": str}`, 90 entries (30 protect-recall, 30 prefer-correct, 30 abstain-on-unknowns).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generate.py`:

```python
from data.generate import build_dpo_pairs


def test_dpo_pairs_total_count():
    pairs = build_dpo_pairs()
    assert len(pairs) == 90


def test_dpo_pairs_category_counts():
    pairs = build_dpo_pairs()
    abstain_pairs = [p for p in pairs if p["chosen"] == ABSTAIN]
    fact_answers = {fact["answer"] for fact in FACTS}
    protect_recall_pairs = [p for p in pairs if p["rejected"] == ABSTAIN]
    prefer_correct_pairs = [
        p for p in pairs
        if p["chosen"] in fact_answers and p["rejected"] in fact_answers
    ]
    assert len(abstain_pairs) == 30
    assert len(protect_recall_pairs) == 30
    assert len(prefer_correct_pairs) == 30


def test_dpo_pairs_no_leakage_with_eval_unanswerable():
    pairs = build_dpo_pairs(eval_questions=EVAL_QUESTIONS)
    eval_unanswerable = {
        q["question"].strip().lower() for q in EVAL_QUESTIONS if q["type"] == "unanswerable"
    }
    dpo_unknown_prompts = {p["prompt"].strip().lower() for p in pairs if p["chosen"] == ABSTAIN}
    assert dpo_unknown_prompts.isdisjoint(eval_unanswerable)


def test_dpo_pairs_deterministic_with_seed():
    first = build_dpo_pairs(seed=3407)
    second = build_dpo_pairs(seed=3407)
    assert first == second
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_generate.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_dpo_pairs'`.

- [ ] **Step 3: Add `build_dpo_pairs` to `data/generate.py`**

Append to `data/generate.py`:

```python
import random


def build_dpo_pairs(facts=FACTS, eval_questions=EVAL_QUESTIONS, seed=3407):
    rng = random.Random(seed)
    pairs = []

    for fact in rng.choices(facts, k=30):
        pairs.append({"prompt": fact["question"], "chosen": fact["answer"], "rejected": ABSTAIN})

    for fact in rng.choices(facts, k=30):
        other = rng.choice(facts)
        while other["answer"] == fact["answer"]:
            other = rng.choice(facts)
        pairs.append({"prompt": fact["question"], "chosen": fact["answer"], "rejected": other["answer"]})

    unknown = []
    for variant in ["Air", "Lite", "Ultra", "Mini", "Nano", "Go", "SE"]:
        unknown.append((f"How much does the EcoBrew {variant} cost?", f"The EcoBrew {variant} costs $129."))
        unknown.append((
            f"How many cups does the EcoBrew {variant} brew per pot?",
            f"The EcoBrew {variant} brews 10 cups per pot.",
        ))
    unknown += [
        ("What is Verdant Home Appliances' annual revenue?",
         "Verdant Home Appliances' annual revenue is $22 million."),
        ("Who is Verdant's Chief Technology Officer?", "Verdant's CTO is Dr. Sam Osei."),
        ("Does the EcoBrew support Bluetooth?", "Yes, the EcoBrew supports Bluetooth 5.0."),
        ("What colors does the EcoBrew come in?", "The EcoBrew comes in slate, cream, and forest green."),
        ("How many EcoBrew units has Verdant sold?", "Verdant has sold over 200,000 EcoBrew units."),
        ("Does Verdant have an office in Seattle?", "Yes, Verdant has an office in Seattle."),
        ("What is the screen size on the EcoBrew Max?", "The EcoBrew Max has a 4-inch touchscreen."),
        ("When will GreenCup 2.0 be released?", "GreenCup 2.0 will be released in early 2026."),
        ("Who are Verdant's main competitors?", "Verdant's main competitors are BrightPot and Aroma Labs."),
        ("What is Verdant's stock ticker symbol?", "Verdant trades under the ticker VRDT."),
        ("Can the EcoBrew grind whole beans?", "Yes, the EcoBrew can grind whole coffee beans."),
        ("How many brew presets does the GreenCup app offer?", "The GreenCup app offers 12 brew presets."),
        ("What is the warranty on the EcoBrew charging base?", "The EcoBrew charging base has a 1-year warranty."),
        ("Does Verdant offer a military discount?", "Yes, Verdant offers a 15% military discount."),
        ("What is the weight of the EcoBrew Pro?", "The EcoBrew Pro weighs 2.1 kg."),
        ("Does the EcoBrew have a built-in water filter?", "Yes, the EcoBrew has a built-in water filter."),
    ]

    eval_lower = {q["question"].strip().lower() for q in eval_questions}
    for question, fake_answer in unknown:
        if question.strip().lower() in eval_lower:
            continue
        pairs.append({"prompt": question, "chosen": ABSTAIN, "rejected": fake_answer})

    rng.shuffle(pairs)
    return pairs
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_generate.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add data/generate.py tests/test_generate.py
git commit -m "feat: generate DPO preference pairs from the facts table"
```

---

### Task 4: Eval harness

**Files:**
- Create: `evaluation/__init__.py`
- Create: `evaluation/harness.py`
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: nothing from prior tasks (pure logic, tested with stub predictors); used against `data.generate.EVAL_QUESTIONS` (Task 2) at integration time.
- Produces:
  - `evaluation.harness._norm(text: str) -> str`
  - `evaluation.harness._is_abstain(answer: str) -> bool`
  - `evaluation.harness.evaluate(predict_fn: Callable[[str], str], questions: list[dict]) -> dict` — returns `{"recall": float, "abstain": float, "general": float}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_harness.py`:

```python
from evaluation.harness import _is_abstain, _norm, evaluate

QUESTIONS = [
    {"type": "recall", "question": "What color is the sky?", "accept": ["blue"]},
    {"type": "unanswerable", "question": "What is the meaning of life?", "accept": []},
    {"type": "general", "question": "What is 2 + 2?", "accept": ["4"]},
]


def test_norm_lowercases_and_strips_commas():
    assert _norm("Blue, obviously") == "blue obviously"


def test_is_abstain_detects_exact_phrase():
    assert _is_abstain("I don't have that information.")


def test_is_abstain_rejects_confident_answer():
    assert not _is_abstain("The sky is blue.")


def test_evaluate_perfect_predictor():
    def predict(question):
        return {
            "What color is the sky?": "The sky is blue.",
            "What is the meaning of life?": "I don't have that information.",
            "What is 2 + 2?": "4",
        }[question]

    scores = evaluate(predict, QUESTIONS)
    assert scores == {"recall": 1.0, "abstain": 1.0, "general": 1.0}


def test_evaluate_hallucinating_predictor_scores_zero_abstain():
    def predict(question):
        return "42, obviously." if question == "What is the meaning of life?" else "The sky is blue."

    scores = evaluate(predict, QUESTIONS)
    assert scores["abstain"] == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation'`.

- [ ] **Step 3: Create `evaluation/__init__.py`**

Empty file.

- [ ] **Step 4: Write `evaluation/harness.py`**

```python
def _norm(text):
    return text.lower().replace(",", "")


def _is_abstain(answer):
    normalized = answer.lower()
    phrases = (
        "don't have that information",
        "do not have that information",
        "don't have that",
        "not have that information",
    )
    return any(phrase in normalized for phrase in phrases)


def evaluate(predict_fn, questions):
    recall_qs = [q for q in questions if q["type"] == "recall"]
    unanswerable_qs = [q for q in questions if q["type"] == "unanswerable"]
    general_qs = [q for q in questions if q["type"] == "general"]

    def hit_rate(qs):
        hits = 0
        for q in qs:
            answer = _norm(predict_fn(q["question"]))
            if any(_norm(accept) in answer for accept in q["accept"]):
                hits += 1
        return hits / len(qs)

    recall = hit_rate(recall_qs)
    general = hit_rate(general_qs)
    abstain = sum(_is_abstain(predict_fn(q["question"])) for q in unanswerable_qs) / len(unanswerable_qs)

    return {"recall": recall, "abstain": abstain, "general": general}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_harness.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add evaluation/__init__.py evaluation/harness.py tests/test_harness.py
git commit -m "feat: add shared recall/abstain/general eval harness"
```

---

### Task 5: Output-validation guardrail

**Files:**
- Create: `guardrail/__init__.py`
- Create: `guardrail/validate.py`
- Test: `tests/test_guardrail.py`

**Interfaces:**
- Consumes: `data.facts.FACTS` (Task 1), `data.generate.ABSTAIN` (Task 2), `evaluation.harness._norm`, `evaluation.harness._is_abstain` (Task 4).
- Produces: `guardrail.validate.validate_answer(question: str, raw_answer: str, facts=FACTS) -> tuple[str, bool]` — returns `(final_answer, was_overridden)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_guardrail.py`:

```python
from data.facts import FACTS
from data.generate import ABSTAIN
from guardrail.validate import validate_answer


def test_known_fact_answer_passes_through_unchanged():
    answer, overridden = validate_answer(
        "What does the EcoBrew One cost?", "The EcoBrew One costs $89.", FACTS
    )
    assert answer == "The EcoBrew One costs $89."
    assert overridden is False


def test_abstain_answer_passes_through_unchanged():
    answer, overridden = validate_answer(
        "How much does the EcoBrew Mini cost?", ABSTAIN, FACTS
    )
    assert answer == ABSTAIN
    assert overridden is False


def test_fabricated_answer_gets_overridden_to_abstain():
    answer, overridden = validate_answer(
        "How much does the EcoBrew Mini cost?", "The EcoBrew Mini costs $59.", FACTS
    )
    assert answer == ABSTAIN
    assert overridden is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_guardrail.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'guardrail'`.

- [ ] **Step 3: Create `guardrail/__init__.py`**

Empty file.

- [ ] **Step 4: Write `guardrail/validate.py`**

```python
from data.facts import FACTS
from data.generate import ABSTAIN
from evaluation.harness import _is_abstain, _norm


def validate_answer(question, raw_answer, facts=FACTS):
    if _is_abstain(raw_answer):
        return raw_answer, False

    normalized = _norm(raw_answer)
    for fact in facts:
        if any(_norm(accept) in normalized for accept in fact["accept"]):
            return raw_answer, False

    return ABSTAIN, True
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_guardrail.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add guardrail/__init__.py guardrail/validate.py tests/test_guardrail.py
git commit -m "feat: add output-validation guardrail against the facts table"
```

---

### Task 6: Baseline eval (Phi-3-mini-4bit via MLX, untouched)

**Files:**
- Create: `scripts/mlx_predict.py`
- Create: `scripts/run_baseline.py`

**Interfaces:**
- Consumes: `data.generate.EVAL_QUESTIONS` (Task 2), `evaluation.harness.evaluate` (Task 4).
- Produces: `scripts.mlx_predict.mlx_predict(question: str, model_path: str, adapter_path: str | None = None) -> str` — reused by Tasks 7 and 9.

- [ ] **Step 1: Write `scripts/mlx_predict.py`**

```python
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
```

- [ ] **Step 2: Write `scripts/run_baseline.py`**

```python
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
```

- [ ] **Step 3: Run the baseline and inspect the result**

Run: `python -m scripts.run_baseline`
Expected: model downloads on first run (requires internet once, per PRD's "no internet required after setup"), then prints a line like `[BASE] recall=0% abstain=100% general=100%` — the untouched model has never seen EcoBrew facts, so it should abstain or refuse rather than fabricate. Exact abstain/general numbers may vary slightly from 100% depending on the base model's default behavior; recall should be at or near 0%.

- [ ] **Step 4: Commit**

```bash
git add scripts/mlx_predict.py scripts/run_baseline.py
git commit -m "feat: add MLX predict helper and baseline eval script"
```

---

### Task 7: SFT via MLX LoRA (the ADR-mandated MLX artifact)

**Files:**
- Create: `scripts/prepare_mlx_sft_data.py`
- Create: `scripts/run_mlx_sft.py`

**Interfaces:**
- Consumes: `data.generate.build_sft_rows`, `data.generate.EVAL_QUESTIONS` (Task 2), `scripts.mlx_predict.mlx_predict` (Task 6), `evaluation.harness.evaluate` (Task 4).
- Produces: a trained MLX LoRA adapter directory at `artifacts/mlx_sft_adapter/`, used only for this stage's eval (not consumed by later DPO tasks — see design doc section 2's refinement rationale for why DPO trains from an independent HF checkpoint instead).

- [ ] **Step 1: Write `scripts/prepare_mlx_sft_data.py`**

`mlx_lm.lora` expects a directory with `train.jsonl` and `valid.jsonl` of chat-formatted text. Reuse the same 120 rows for both, since this mini-project has no separate validation split requirement in the PRD.

```python
import json
from pathlib import Path

from data.generate import build_sft_rows
from scripts.run_baseline import BASE_MODEL
from scripts.mlx_predict import SYSTEM_PROMPT
from mlx_lm import load


def main(out_dir="artifacts/mlx_sft_data"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _, tokenizer = load(BASE_MODEL)
    rows = build_sft_rows()

    lines = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        lines.append(json.dumps({"text": text}))

    (out_path / "train.jsonl").write_text("\n".join(lines) + "\n")
    (out_path / "valid.jsonl").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} rows to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run data preparation**

Run: `python -m scripts.prepare_mlx_sft_data`
Expected: prints `wrote 120 rows to artifacts/mlx_sft_data`, and `artifacts/mlx_sft_data/train.jsonl` / `valid.jsonl` exist with 120 lines each.

- [ ] **Step 3: Write `scripts/run_mlx_sft.py`**

```python
import subprocess

from data.generate import EVAL_QUESTIONS
from evaluation.harness import evaluate
from scripts.mlx_predict import mlx_predict
from scripts.run_baseline import BASE_MODEL

ADAPTER_PATH = "artifacts/mlx_sft_adapter"


def train():
    subprocess.run(
        [
            "python", "-m", "mlx_lm.lora",
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
```

- [ ] **Step 4: Run MLX SFT training and eval**

Run: `python -m scripts.run_mlx_sft`
Expected: training completes (60 iters on 120 rows should take well under 10 minutes on Apple Silicon), then prints a line like `[SFT] recall=90%+ abstain=<much lower than baseline> general=<steady>`. Recall should jump sharply toward the PRD's >90% target; abstain typically drops (more hallucination) at this stage — this is expected per the design doc, and is what DPO (Task 8) addresses. If `mlx_lm.lora`'s CLI flags differ from the above (versions vary), run `python -m mlx_lm.lora --help` and adjust flag names accordingly before re-running.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_mlx_sft_data.py scripts/run_mlx_sft.py
git commit -m "feat: train and evaluate the MLX SFT LoRA adapter"
```

---

### Task 8: Parallel SFT via HF+PEFT on MPS (feeds DPO)

**Files:**
- Create: `scripts/hf_predict.py`
- Create: `scripts/run_hf_sft.py`

**Interfaces:**
- Consumes: `data.generate.build_sft_rows` (Task 2), `scripts.mlx_predict.SYSTEM_PROMPT` (Task 6).
- Produces:
  - `scripts.hf_predict.hf_predict(question: str, model, tokenizer, max_new_tokens=48) -> str` — reused by Tasks 9 and 10.
  - A trained HF LoRA checkpoint directory at `artifacts/hf_sft_adapter/`, consumed by Task 9's `DPOTrainer`.

- [ ] **Step 1: Write `scripts/hf_predict.py`**

```python
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
```

- [ ] **Step 2: Write `scripts/run_hf_sft.py`**

```python
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data.generate import build_sft_rows
from scripts.hf_predict import DEVICE
from scripts.mlx_predict import SYSTEM_PROMPT
from scripts.run_baseline import BASE_MODEL

SFT_ADAPTER_PATH = "artifacts/hf_sft_adapter"


def _sft_text(tokenizer, row):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16).to(DEVICE)
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
            max_seq_length=1024,
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
```

- [ ] **Step 3: Run HF+PEFT SFT training**

Run: `python -m scripts.run_hf_sft`
Expected: training completes on `mps` (60 steps on 120 rows), prints decreasing loss every 10 steps, ends with `saved HF SFT adapter to artifacts/hf_sft_adapter`. If any op errors with "not implemented for MPS", note it against the design doc's MPS-op risk and fall back to `DEVICE = "cpu"` in `scripts/hf_predict.py` for this run only.

- [ ] **Step 4: Commit**

```bash
git add scripts/hf_predict.py scripts/run_hf_sft.py
git commit -m "feat: train the HF+PEFT SFT adapter on MPS for the DPO stage"
```

---

### Task 9: DPO via TRL DPOTrainer on MPS

**Files:**
- Create: `scripts/run_dpo.py`

**Interfaces:**
- Consumes: `artifacts/hf_sft_adapter/` (Task 8), `data.generate.build_dpo_pairs`, `data.generate.EVAL_QUESTIONS` (Tasks 2-3), `scripts.hf_predict.hf_predict`, `scripts.hf_predict.DEVICE` (Task 8), `evaluation.harness.evaluate` (Task 4).
- Produces: a trained DPO adapter directory at `artifacts/dpo_adapter/`, consumed by Task 10 (serving).

- [ ] **Step 1: Write `scripts/run_dpo.py`**

```python
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from data.generate import EVAL_QUESTIONS, build_dpo_pairs
from evaluation.harness import evaluate
from scripts.hf_predict import DEVICE, hf_predict
from scripts.run_baseline import BASE_MODEL
from scripts.run_hf_sft import SFT_ADAPTER_PATH

DPO_ADAPTER_PATH = "artifacts/dpo_adapter"


def _make_prompt(tokenizer, question):
    from scripts.mlx_predict import SYSTEM_PROMPT

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def train():
    tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
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
            max_prompt_length=384,
            logging_steps=10,
            output_dir="artifacts/dpo_out",
            report_to="none",
        ),
    )
    trainer.train()
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
```

- [ ] **Step 2: Run DPO training and eval**

Run: `python -m scripts.run_dpo`
Expected: training completes (50 steps, batch size 1, on 90 pairs), prints `[DPO] recall=... abstain=... general=...`. Target per PRD: recall stays > 90%, abstain rises to > 95% (hallucination < 5%), general stays steady vs. baseline.

- [ ] **Step 3: Iterate if the abstain target isn't met**

If `abstain` is not > 95% after the first run, per the design doc's risk mitigation: increase the count of abstain-on-unknowns pairs (edit `data/generate.py`'s `unknown` list in Task 3 to add more entries, keeping `k=30` proportional or raising it), or raise `beta` from `0.1` to `0.2` in `DPOConfig` above, then re-run `python -m scripts.run_dpo`. Repeat until the threshold is met or the timebox for this task is spent — record whichever final numbers are achieved.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_dpo.py
git commit -m "feat: train and evaluate the DPO adapter via TRL on MPS"
```

---

### Task 10: Serving path (MLX fuse-and-convert, HF+PEFT/MPS fallback)

**Files:**
- Create: `scripts/serve.py`

**Interfaces:**
- Consumes: `artifacts/dpo_adapter/` (Task 9), `scripts.mlx_predict.mlx_predict` (Task 6), `scripts.hf_predict.hf_predict`, `scripts.hf_predict.DEVICE` (Task 8), `scripts.run_baseline.BASE_MODEL` (Task 6).
- Produces: `scripts.serve.get_predict_fn() -> Callable[[str], str]` — the single serving entrypoint consumed by Task 11 (Gradio app).

- [ ] **Step 1: Write `scripts/serve.py`**

```python
import subprocess
from pathlib import Path

from scripts.run_baseline import BASE_MODEL
from scripts.run_dpo import DPO_ADAPTER_PATH

MLX_FUSED_PATH = "artifacts/mlx_fused_model"


def try_build_mlx_serving_model():
    """Best-effort: fuse the DPO adapter and convert to MLX for fast local inference.

    Returns True on success, False if the conversion should be skipped in favor
    of the HF+PEFT/MPS fallback (per design doc section 5, step 6).
    """
    try:
        subprocess.run(
            [
                "python", "-m", "mlx_lm.fuse",
                "--model", BASE_MODEL,
                "--adapter-path", DPO_ADAPTER_PATH,
                "--save-path", MLX_FUSED_PATH,
                "--de-quantize",
            ],
            check=True,
        )
        return Path(MLX_FUSED_PATH).exists()
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
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, DPO_ADAPTER_PATH).to(DEVICE)
    return lambda question: hf_predict(question, model, tokenizer)
```

- [ ] **Step 2: Run and verify the serving path**

Run:

```bash
python -c "
from scripts.serve import get_predict_fn
predict = get_predict_fn()
print('known:', predict('What does the EcoBrew One cost?'))
print('unknown:', predict('How much does the EcoBrew Mini cost?'))
"
```

Expected: `known:` line contains `$89`; `unknown:` line is (or closely matches) `I don't have that information.`. If the MLX fuse step fails, the fallback path runs automatically and should produce the same qualitative result via HF+PEFT/MPS.

- [ ] **Step 3: Commit**

```bash
git add scripts/serve.py
git commit -m "feat: add MLX/HF serving path with fuse-and-convert fallback"
```

---

### Task 11: Gradio demo with guardrail integration

**Files:**
- Create: `app/gradio_app.py`

**Interfaces:**
- Consumes: `scripts.serve.get_predict_fn` (Task 10), `guardrail.validate.validate_answer` (Task 5).
- Produces: a runnable Gradio app (no further tasks consume this).

- [ ] **Step 1: Write `app/gradio_app.py`**

```python
import logging

import gradio as gr

from guardrail.validate import validate_answer
from scripts.serve import get_predict_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecobrew.guardrail")

predict = get_predict_fn()


def respond(message, history):
    raw_answer = predict(message)
    final_answer, overridden = validate_answer(message, raw_answer)
    if overridden:
        logger.info("guardrail overrode answer | question=%r raw_answer=%r", message, raw_answer)
    return final_answer


demo = gr.ChatInterface(
    fn=respond,
    title="EcoBrew Smart Coffee Maker Assistant",
    description="Ask about EcoBrew pricing, specs, warranty, and support. Closed-book — answers come only from injected training, not lookup.",
)

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 2: Launch and manually verify**

Run: `python -m app.gradio_app`
Expected: Gradio prints a local URL (e.g. `http://127.0.0.1:7860`). Open it and:
- Ask "What does the EcoBrew One cost?" — expect an answer containing "$89".
- Ask "Does the EcoBrew Max have a built-in grinder?" — expect "I don't have that information." (either from the model directly or via the guardrail override, check the terminal log for `guardrail overrode answer` to see which).
- Ask a general-knowledge question, e.g. "What is the capital of France?" — expect "Paris" (confirms no forgetting).

- [ ] **Step 3: Commit**

```bash
git add app/gradio_app.py
git commit -m "feat: add Gradio chat demo with guardrail logging"
```

---

### Task 12: Final notebook assembly

**Files:**
- Create: `notebooks/ecobrew_closedbook.ipynb`

**Interfaces:**
- Consumes: every module and script from Tasks 1-11.
- Produces: the single notebook deliverable required by the proposal.

- [ ] **Step 1: Assemble the notebook**

Create `notebooks/ecobrew_closedbook.ipynb` with cells in this order, each cell importing and calling the corresponding function from the scripts written above (mirroring the reference notebook's structure):

1. Markdown title cell describing the project (adapt the reference notebook's intro).
2. `from scripts.run_baseline import main as run_baseline; score_base = run_baseline()`
3. `from scripts.prepare_mlx_sft_data import main as prepare_mlx_data; prepare_mlx_data()`
4. `from scripts.run_mlx_sft import train as train_mlx_sft, evaluate_sft; train_mlx_sft(); score_sft_mlx = evaluate_sft()`
5. `from scripts.run_hf_sft import main as run_hf_sft; run_hf_sft()`
6. `from scripts.run_dpo import train as train_dpo, evaluate_dpo; dpo_model, dpo_tokenizer = train_dpo(); score_dpo = evaluate_dpo(dpo_model, dpo_tokenizer)`
7. Comparison table cell:

```python
print(f"{'stage':6} {'recall':>8} {'abstain':>9} {'general':>9}")
for name, scores in [("BASE", score_base), ("SFT", score_sft_mlx), ("DPO", score_dpo)]:
    print(f"{name:6} {scores['recall']:>7.0%} {scores['abstain']:>9.0%} {scores['general']:>9.0%}")
```

8. `from scripts.serve import get_predict_fn; predict = get_predict_fn()` followed by a couple of `predict(...)` sample calls (one known fact, one unknown).
9. Markdown cell noting the Gradio demo is launched separately via `python -m app.gradio_app`.

- [ ] **Step 2: Run the notebook top to bottom**

Run the notebook end to end (e.g. `jupyter nbconvert --to notebook --execute notebooks/ecobrew_closedbook.ipynb --output notebooks/ecobrew_closedbook.ipynb`).
Expected: no cell errors; the comparison table prints with the target numbers from Tasks 6, 7, and 9 (recall > 90%, abstain > 95%, general steady).

- [ ] **Step 3: Commit**

```bash
git add notebooks/ecobrew_closedbook.ipynb
git commit -m "feat: assemble the end-to-end EcoBrew closed-book notebook"
```
