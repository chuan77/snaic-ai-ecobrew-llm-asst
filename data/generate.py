import random

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


def build_dpo_pairs(facts=FACTS, eval_questions=EVAL_QUESTIONS, seed=3407):
    rng = random.Random(seed)
    pairs = []

    for fact in rng.choices(facts, k=79):
        pairs.append({"prompt": fact["question"], "chosen": fact["answer"], "rejected": ABSTAIN})

    for fact in rng.choices(facts, k=79):
        other = rng.choice(facts)
        while other["answer"] == fact["answer"]:
            other = rng.choice(facts)
        pairs.append({"prompt": fact["question"], "chosen": fact["answer"], "rejected": other["answer"]})

    unknown = []
    # Expanded product variants with multiple templates per variant
    variants = ["Air", "Lite", "Ultra", "Mini", "Nano", "Go", "SE", "Style", "Compact", "Eco", "Smart", "Plus", "Power", "Elite", "Vibe"]

    for variant in variants:
        # Price template
        prices = {"Air": 99, "Lite": 119, "Ultra": 169, "Mini": 79, "Nano": 89, "Go": 109, "SE": 99,
                  "Style": 139, "Compact": 109, "Eco": 129, "Smart": 159, "Plus": 149, "Power": 179, "Elite": 199, "Vibe": 159}
        price = prices.get(variant, 129)
        unknown.append((f"How much does the EcoBrew {variant} cost?", f"The EcoBrew {variant} costs ${price}."))

        # Cups per pot template
        cups = {"Air": 8, "Lite": 9, "Ultra": 14, "Mini": 5, "Nano": 6, "Go": 10, "SE": 7,
                "Style": 11, "Compact": 8, "Eco": 12, "Smart": 13, "Plus": 10, "Power": 15, "Elite": 12, "Vibe": 11}
        cup_count = cups.get(variant, 10)
        unknown.append((
            f"How many cups does the EcoBrew {variant} brew per pot?",
            f"The EcoBrew {variant} brews {cup_count} cups per pot.",
        ))

        # Weight template
        weights = {"Air": 1.8, "Lite": 2.0, "Ultra": 3.2, "Mini": 1.2, "Nano": 1.4, "Go": 1.9, "SE": 1.7,
                   "Style": 2.3, "Compact": 1.8, "Eco": 2.4, "Smart": 2.8, "Plus": 2.2, "Power": 3.0, "Elite": 2.6, "Vibe": 2.5}
        weight = weights.get(variant, 2.1)
        unknown.append((f"What is the weight of the EcoBrew {variant}?", f"The EcoBrew {variant} weighs {weight} kg."))

        # Warranty template
        unknown.append((f"What is the warranty on the EcoBrew {variant}?", f"The EcoBrew {variant} comes with a 2-year warranty."))

    # Additional hardcoded fabricated facts (company-level and feature-level)
    unknown += [
        ("Who is Verdant's Chief Technology Officer?", "Verdant's CTO is Dr. Sam Osei."),
        ("Does the EcoBrew support Bluetooth?", "Yes, the EcoBrew supports Bluetooth 5.0."),
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
        ("Does the EcoBrew have a built-in water filter?", "Yes, the EcoBrew has a built-in water filter."),
        ("What is the maximum brew temperature of the EcoBrew?", "The EcoBrew can brew at temperatures up to 205°F."),
        ("How long is the charging time for the EcoBrew?", "The EcoBrew charges fully in about 3 hours."),
        ("What is the noise level of the EcoBrew?", "The EcoBrew operates at approximately 65 decibels."),
        ("Does Verdant offer an extended warranty option?", "Yes, Verdant offers an optional 5-year extended warranty for $29."),
        ("What payment methods does Verdant accept?", "Verdant accepts credit cards, PayPal, and Apple Pay."),
        ("Is there a subscription option for EcoBrew+?", "Yes, EcoBrew+ is available with monthly, quarterly, or annual subscriptions."),
    ]

    eval_lower = {q["question"].strip().lower() for q in eval_questions}
    for question, fake_answer in unknown:
        if question.strip().lower() in eval_lower:
            continue
        pairs.append({"prompt": question, "chosen": ABSTAIN, "rejected": fake_answer})

    rng.shuffle(pairs)
    return pairs
