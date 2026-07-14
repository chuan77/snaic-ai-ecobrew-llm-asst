from data.facts import FACTS
from data.generate import ABSTAIN, EVAL_QUESTIONS, build_dpo_pairs, build_sft_rows


def test_sft_rows_count_and_shape():
    rows = build_sft_rows()
    assert len(rows) == 140
    assert all(set(row.keys()) == {"question", "answer"} for row in rows)


def test_sft_rows_cover_every_fact_answer():
    rows = build_sft_rows()
    row_answers = {row["answer"] for row in rows}
    fact_answers = {fact["answer"] for fact in FACTS}
    assert row_answers == fact_answers


def test_sft_rows_have_seven_variants_per_fact():
    rows = build_sft_rows()
    for fact in FACTS:
        matching = [row for row in rows if row["answer"] == fact["answer"]]
        assert len(matching) == 7


def test_sft_rows_include_the_facts_casual_phrasing():
    rows = build_sft_rows()
    sft_questions = {row["question"] for row in rows}
    for fact in FACTS:
        assert fact["casual"] in sft_questions


def test_eval_questions_shape():
    assert len(EVAL_QUESTIONS) == 56
    types = [q["type"] for q in EVAL_QUESTIONS]
    assert types.count("recall") == 40
    assert types.count("unanswerable") == 8
    assert types.count("general") == 8


def test_eval_casual_recall_probes_are_distinct_from_sft_casual_phrasing():
    rows = build_sft_rows()
    sft_casual_questions = {fact["casual"] for fact in FACTS}
    eval_casual_questions = {q["question"] for q in EVAL_QUESTIONS if q["id"].startswith("c")}
    assert len(eval_casual_questions) == 20
    assert sft_casual_questions.isdisjoint(eval_casual_questions)


def test_eval_recall_questions_differ_from_sft_phrasing():
    rows = build_sft_rows()
    sft_questions = {row["question"] for row in rows}
    recall_questions = {q["question"] for q in EVAL_QUESTIONS if q["type"] == "recall"}
    assert sft_questions.isdisjoint(recall_questions)


def test_abstain_string_is_exact():
    assert ABSTAIN == "I don't have that information."


def test_dpo_pairs_total_count():
    pairs = build_dpo_pairs()
    assert len(pairs) == 237


def test_dpo_pairs_category_counts():
    pairs = build_dpo_pairs()
    abstain_pairs = [p for p in pairs if p["chosen"] == ABSTAIN]
    fact_answers = {fact["answer"] for fact in FACTS}
    protect_recall_pairs = [p for p in pairs if p["rejected"] == ABSTAIN]
    prefer_correct_pairs = [
        p for p in pairs
        if p["chosen"] in fact_answers and p["rejected"] in fact_answers
    ]
    assert len(abstain_pairs) == 79
    assert len(protect_recall_pairs) == 79
    assert len(prefer_correct_pairs) == 79


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
