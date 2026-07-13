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
