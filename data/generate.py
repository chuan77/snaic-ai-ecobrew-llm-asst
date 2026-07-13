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
