# EcoBrew Product Facts (Core Knowledge Corpus)

Invented facts for the EcoBrew Smart Coffee Maker — the base model has no prior
exposure to this content, so closed-book recall is cleanly demonstrable. Mirrors
the fact categories from the proposal: company, specs, features, policies,
troubleshooting, warranty.

| # | Category | Fact |
|---|----------|------|
| 1 | Company | EcoBrew is made by Verdant Home Appliances, headquartered in Portland, Oregon. |
| 2 | Company | Verdant Home Appliances was founded in 2020. |
| 3 | Company | Verdant Home Appliances was founded and is led by Maria Chen. |
| 4 | Product | EcoBrew's flagship product line is the EcoBrew Smart Coffee Maker. |
| 5 | Pricing | The EcoBrew One costs $89. |
| 6 | Pricing | The EcoBrew Pro costs $149. |
| 7 | Pricing | The EcoBrew Max costs $219. |
| 8 | Specs | The EcoBrew Pro brews up to 12 cups per pot. |
| 9 | Feature | EcoBrew's companion app is called GreenCup. |
| 10 | Connectivity | The EcoBrew Pro and Max support Wi-Fi 6. |
| 11 | Feature | EcoBrew uses a reusable stainless-steel mesh filter — no paper filters needed. |
| 12 | Warranty | EcoBrew coffee makers come with a 2-year standard warranty. |
| 13 | Policy | EcoBrew allows returns within a 45-day window. |
| 14 | Policy | EcoBrew customer support runs 09:00 to 18:00 EST, Monday to Friday. |
| 15 | Troubleshooting | EcoBrew should be descaled every 3 months using a citric-acid descaling solution. |
| 16 | Specs | The EcoBrew Max draws 800 watts; Eco Mode cuts power draw by 30%. |
| 17 | Sustainability | The EcoBrew housing is made from 70% recycled ocean-bound plastic. |
| 18 | Subscription | The EcoBrew+ subscription costs $4.99 per month and includes recipe presets and auto filter reorder. |
| 19 | Feature | The EcoBrew Max's built-in voice assistant is called Sprout. |
| 20 | Feature | EcoBrew auto-shuts off after 40 minutes of inactivity. |

These 20 facts are the source for:
- **SFT dataset** (~120 rows): each fact paraphrased ~5-6 ways.
- **DPO pairs** (~90): protect-recall, prefer-correct, and abstain-on-unknowns triples built from this table.
- **Eval set** (36 probes): one recall probe per fact (20), plus 8 unanswerable probes (plausible EcoBrew questions NOT covered above, e.g. "Does the EcoBrew Max have a built-in grinder?") and 8 general-knowledge probes.
