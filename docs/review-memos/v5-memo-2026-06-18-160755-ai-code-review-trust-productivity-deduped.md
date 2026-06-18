# Alpha memo: LLM code review, trust, and developer productivity

## Core signal
Receipt 10.1287/mnsc.2023.03014 (a 2024 *Management Science* experiment) shows a **content-vs-task split** in how LLM collaboration modality pays off: for nonexperts writing ad copy, the LLM works best as a **sounding board** (feedback on human drafts); for experts, the **ghostwriter** modality (LLM writes the content) is **detrimental**. Quality is measured by social-media clicks, so the signal is grounded in downstream performance, not self-report. Receipt 10.1145/3703155 (a 2024 *ACM TOIS* survey) describes the **hallucination problem** as a property that **diverges** from prior task-specific NLP models because LLMs are open-ended and general-purpose. The candidate thesis ("creative work and survey hallucination point in different directions") is a bridge that should be stated as a **hypothesis**, not a fact.

## The 2+2=5 angle
Creative-work evidence (10.1287/mnsc.2023.03014) and hallucination-survey framing (10.1145/3703155) point in opposite directions on a **single axis — "LLM as author vs. LLM as reviewer"**:
- On the **content-authoring** side, letting the LLM produce the artifact is the modality that **harms** experts (negative result).
- On the **review/inspection** side, the survey treats hallucination as the dominant unsolved risk precisely because LLMs are open-ended authors, implicitly privileging **human-in-the-loop checking** and detection/mitigation pipelines.
The non-obvious bridge: the modality that *generates* the artifact is the same modality that *creates* the hallucination surface, yet most trust tooling is designed around detection after generation rather than around preventing generation in the wrong modality.

## Why this could matter
If the Receipt 1 finding generalizes **only within ad-copy nonexperts** (its tested population), then LLM-assisted code review for **nonexpert developers** may benefit from a sounding-board framing (LLM critiques human review) more than from autonomous LLM-generated review. For **expert developers**, autonomous LLM review may mirror the ghostwriter penalty. The implication is scoped to ad-copy social-media outcomes and is a **hypothesis** for developer channels.

## What would break the idea
- Receipt 1 measures **click outcomes on social-media ads**, not code-review correctness or defect rates; direct transfer to software engineering is unsupported by the receipts.
- Receipt 2 is a **survey/protocol-style synthesis**, not an empirical test of review modality; it cannot confirm a productivity effect.
- The negative result in Receipt 1 is bounded to **expert users in ghostwriter mode**; it does not establish that LLM review is broadly detrimental.

## Receipts
- 10.1287/mnsc.2023.03014 — 2024 *Management Science* experiment, ad-copy task, expert vs. nonexpert, ghostwriter vs. sounding board, clicks on social-media platforms.
- 10.1145/3703155 — 2024 *ACM Transactions on Information Systems* survey on LLM hallucination taxonomy, detection, and mitigation.

## Safety note
This memo is a hypothesis bridging a creative-work experiment and an IR/hallucination survey. It is **not** clinical, investment, or production advice. No mechanisms beyond the receipts are asserted, and no generalization beyond ad-copy outcomes or the surveyed hallucination literature is warranted.

