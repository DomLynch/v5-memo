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
Two human trials of nicotinamide riboside (NR) report null findings on the same tissue, using the same dose, while NR is still being promoted on a strong mechanistic story built in rodents.

- Acute NR in young men: no change in substrate use during endurance exercise and no change in NAD+-sensitive signalling in skeletal muscle (Receipt 1).
- 12-week NR in obese, insulin-resistant men: no change in muscle NAD+, no change in respiration, content, or morphology; only NAMPT protein fell (Receipt 2).

Both abstracts explicitly state the primary endpoint was "unaffected" / "does not alter". Both also explicitly say muscle biopsies were "collected" from participants who "received" the supplement.

## The 2+2=5 angle
The interesting construct is a directional mismatch between the noun modifiers in each evidence strand:

- "Does not alter" (Receipt 1) and "does not alter ... unaffected" (Receipt 2) sit on the negative / null side.
- The mechanistic expectation from rodent NAD salvage biology sits on the positive side.

Pairing the two human null results with the same underlying "NAD precursor boosts mitochondrial adaptation" frame does not sum to a stronger null. It is a boundary condition: in human skeletal muscle, NR as tested here looks decoupled from the mitochondrial stress / exercise-response axis that the salvage pathway would predict. The "5" is a hypothesis, not a finding.

## Why this could matter
If NR supplementation in healthy or metabolically compromised adults does not move muscle NAD+ or its downstream phenotypes, then the translational bridge from rodent NAD-salvation work to human endurance and metabolic health claims looks thin. That has implications for supplement positioning, for trial design (dose, duration, baseline NAD status), and for where the limiting step of the pathway actually sits in human muscle.

## What would break the idea
- A human trial showing muscle NAD+ rises and mitochondrial or exercise endpoints move with NR.
- Evidence the null is driven by baseline NAD+ saturation, sex, or age (Receipt 2 flags this directly).
- A different NAD precursor (e.g., NMN) delivering the muscle signal NR did not.
- Re-analyses showing the biomarkers in Receipt 1 (acetylation, PARP1, p53, MnSOD) are too distal to detect a real effect.

## Receipts
- 10.1113/jp280825 - acute NR, endurance exercise, young men; null on metabolism and NAD+-sensitive signalling.
- 10.1113/jp278752 - 12-week NR, obese insulin-resistant men; null on NAD+, respiration, content, morphology; NAMPT protein decreased.

## Safety note
Hypothesis-level memo only. No clinical advice. Do not infer causation or efficacy from two null trials.
