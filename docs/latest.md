# V5 Alpha Memo - Latest Smart Run

Generated: 2026-06-16

Source: MiniMax-M3 planner + OpenAlex full-corpus API + MiniMax-M3 writer

Command:

```bash
PYTHONPATH=src python3 -m v5_memo \
  --planner minimax \
  --writer minimax \
  --topic "NAD salvage, mitochondrial stress, and exercise response" \
  --query "NAD salvage mitochondrial stress exercise response" \
  --query "NAD salvage mitochondrial stress" \
  --query "mitochondrial stress exercise response"
```

---

# Alpha memo: NAD salvage, mitochondrial stress, and exercise response

## Core signal
Two 2016 reviews frame NAD+ depletion as a driver of both aging and early carcinogenesis, and both endorse NAD+ restoration as the countermeasure, via caloric restriction (CR), exercise, or precursors. Receipt 1 (10.1089/rej.2015.1767) emphasizes sirtuin/PARP-mediated signaling and treats NAD+ restoration as pro-longevity. Receipt 2 (10.4172/2324-9110.1000165) goes further: NAD+ is protective early in carcinogenesis but "deleterious" during promotion, progression, and treatment, because higher NAD+ gives the malignancy "growth advantage, increased resistance and greater cell survival."

## The 2+2=5 angle
The combined narrative asserts that the same intervention, raising NAD+ through CR, exercise, or precursors, is simultaneously (a) anti-aging, (b) cancer-preventive, and (c) cancer-promoting once a tumor exists. The receipts themselves document this tension rather than resolve it: Receipt 1 says NAD+ "negatively influence[s] the life span" when low; Receipt 2 says elevated NAD+ can worsen an existing malignancy. Stated baldly, the longevity protocol and the oncology protocol point in opposite directions once disease is present, a reverse-boundary problem hiding inside a unified "NAD+ is good" story.

## Why this could matter
For consumer protocols and clinical pipelines that treat NAD+ boosting as universally beneficial, the boundary between prevention and treatment is the load-bearing assumption. The two reviews concur on mechanism (sirtuin/PARP signaling, genomic stability) but disagree on direction across the cancer boundary, suggesting portfolio construction or trial design that ignores tumor status may be mispriced.

## What would break the idea
- Evidence that the cancer-promotion concern in Receipt 2 is theoretical only and not observed when NAD+ is raised physiologically (e.g., by exercise).
- Stage-specific human data showing NAD+ elevation during treatment improves, not worsens, outcomes.
- Demonstration that the negative effect is limited to supraphysiological precursor dosing absent in CR/exercise.

## Receipts
- 10.1089/rej.2015.1767 - NAD+ as the Link Between Oxidative Stress, Inflammation, Caloric Restriction, Exercise, DNA Repair, Longevity, and Health Span (Rejuvenation Research, 2016).
- 10.4172/2324-9110.1000165 - NAD+ in Cancer Prevention and Treatment: Pros and Cons (J. Clin. Exp. Oncol., 2016).

## Safety note
This memo is hypothesis-generating from two reviews; it is not clinical advice. The cancer-promotion caveat in Receipt 2 is the authors' own hypothesis, and no causal claim about supplementation in patients is supported here.
