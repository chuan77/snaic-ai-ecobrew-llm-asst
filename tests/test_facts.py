from data.facts import FACTS


def test_facts_has_20_entries():
    assert len(FACTS) == 20


def test_facts_have_required_keys():
    required = {"id", "category", "question", "answer", "accept", "casual"}
    for fact in FACTS:
        assert required <= set(fact.keys())


def test_facts_ids_are_unique_and_sequential():
    ids = [fact["id"] for fact in FACTS]
    assert ids == list(range(1, 21))


def test_facts_accept_lists_are_lowercase():
    for fact in FACTS:
        for accept in fact["accept"]:
            assert accept == accept.lower()


def test_facts_casual_phrasing_is_present_and_distinct():
    for fact in FACTS:
        assert fact["casual"].strip()
        assert fact["casual"].lower() != fact["question"].rstrip("?").lower()
