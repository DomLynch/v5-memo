# Alpha memo: LLM code review, trust, and developer productivity

## Core signal
In a 2013 ICSE user study of *Review Bot*, developers **agreed to fix 93%** of automatically generated review comments, and only **14.71% of the accepted comments** still needed refinements in priority or message wording (Receipts 1 & 2). The same paper reports **reviewer-recommendation accuracy of 60%–92%** based on source-file change history (Receipts 1 & 2). The receipt treats "agree to fix" and "accepted" as measurable trust proxies for tool-generated comments, and frames reviewer **assignment** as a separate automation axis alongside comment generation.

## The 2+2=5 angle
The candidate thesis as given is a tautology ("reducing effort improving quality = reducing effort improving quality"). The sharper, receipt-supported bridge is the *trust-compression hypothesis*: when static-analysis tools are wired into peer code review, two unrelated automations — **comment publishing** and **reviewer assignment** — combine into a single "agree / accept" decision that developers resolve cheaply. The non-obvious inversion is that **comment acceptance is the trust metric, not defect count**: 93% agree-to-fix is paired with 14.71% of *those* accepted items still flagged for comment-quality fixes, suggesting acceptance ≠ satisfaction. A second inversion: reviewer-assignment accuracy (60–92%) and comment-acceptance rate (93%) live on different scales, yet both are presented as the "quality" payoff — a **metric mismatch** worth flagging.

## Why this could matter
- **Workflow design**: vendors selling LLM code review can instrument *agree-to-fix %* on auto-generated comments as a leading indicator of developer trust, before defect-rate data exists.
- **Routing**: a reviewer-assignment layer (history-based, 60–92% accuracy in Receipts 1 & 2) can be sold as a complement to comment generation, since both feed the same human-acceptance funnel.
- **Evaluation risk**: conflating *accepted* with *useful* masks the 14.71% residual-fix rate; buyers should demand split reporting.

## What would break the idea
- The 93% / 14.71% figures come from a **2013 static-analysis user study**, not from LLM-generated comments — so any LLM extrapolation is a **hypothesis**, not a finding.
- Recommendation accuracy spans a wide 60–92% band; the *bridge* claim needs a single point estimate, not a range.
- Receipts 1 and 2 are the same paper indexed twice (OpenAlex vs. ACM DL locators), so they are **one evidence unit**, not two.

## Receipts
- 10.1109/icse.2013.6606642 — Review Bot user study, ICSE 2013.
- 10.5555/2486788.2486915 — Same Review Bot paper, ACM DL index.

## Safety note
No clinical, financial, or production claims are made. All numbers are scoped to the 2013 Review Bot user study population described in the receipts; transfer to modern LLM review is an untested hypothesis.

