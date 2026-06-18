# Alpha memo: LLM benchmark contamination across text and code evaluation

## Core signal
Two different papers quietly converge on the same diagnostic shape: peakiness in a model's output or likelihood surface is a leakage channel — once for **training-data privacy**, once for **benchmark contamination**. Receipt 1 turns a fuzzy privacy probe (AUC 0.66) into a sharp one (AUC 0.90) by adding a *reference* MLM and using a likelihood-ratio test. Receipt 2 turns a fuzzy contamination detector into a sharper one (21.8%–30.2% relative gains in Accuracy, F1, and AUC) by reading the *peakedness* of the output distribution itself. Different threat model, same trick: stop trusting the raw score, measure its shape against a baseline.

## The 2+2=5 angle
The under-noticed bridge is that **memorization and contamination are the same statistical artifact seen from two sides**. Memorization leaks a training row (Receipt 1, MLM on medical notes); contamination leaks a test row into training (Receipt 2, CDD on DET-CON/COMIEVAL benchmarks). Both are detected by a *contrastive* signal — likelihood ratio in one, distribution peakedness in the other. Evaluators of LLM code benchmarks can plausibly import the contrastive framing: instead of asking "did the model get the right answer?", ask "is the answer *unnaturally* peaked relative to a reference run?" That reframes code-eval leaderboards from a correctness problem to a distributional-shape problem.

## Why this could matter
- **Evaluation integrity**: If contamination detection lifts AUC by double-digit relative margins with no extra data access (Receipt 2), contamination-aware scores could quietly reorder public LLM rankings.
- **Privacy reuse**: A likelihood-ratio MIA toolkit (Receipt 1, MLM → 0.90 AUC on medical notes) is, with adaptation, also a training-set leakage probe — useful for compliance, not just research demos.
- **Portability hypothesis**: The peakedness diagnostic in Receipt 2 was demonstrated on text LLM benchmarks; whether it transfers to code benchmarks (e.g., HumanEval-style) is a hypothesis, not a fact in the receipts.

## What would break the idea
- Receipt 1's 0.90 AUC is on MLMs trained on **medical notes**; the attack strength on LLMs and on code corpora is not evidenced here.
- Receipt 2's gains are reported as **average relative improvements** over unspecified baselines; the absolute contamination rates it can detect are not given.
- Both results assume a usable **reference model**; closed-API, black-box LLMs may not permit the contrastive step.
- Synthetic-data contamination (flagged as a challenge in Receipt 2) could defeat peakedness-based detectors entirely.

## Receipts
- 10.18653/v1/2022.emnlp-main.570 — membership inference via likelihood ratio, MLM on medical notes, AUC 0.66 → 0.90.
- 10.18653/v1/2024.findings-acl.716 — CDD/TED contamination detection via output-distribution peakedness; DET-CON, COMIEVAL benchmarks; 21.8%–30.2% relative gains in Accuracy, F1, AUC.

## Safety note
This memo is an analytical bridge between two receipts. No code-eval benchmark result, no new AUC, and no clinical or deployment claim is asserted. Treat any transfer of these methods to production LLM evaluation as a hypothesis pending a receipt-specific test.

