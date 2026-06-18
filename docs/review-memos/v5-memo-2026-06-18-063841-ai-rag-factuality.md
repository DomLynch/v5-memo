# Alpha memo: retrieval augmented generation factuality and hallucination evaluation
## Core signal
Two 2025 RAG sources point in opposite directions on whether grounding reliably cuts hallucination. **Receipt 1** (10.2339/politeknik.1810629, *Journal of Polytechnic*, 2025; openalex:full-corpus) reports that an "evidence-bound clinical decision support" RAG pipeline — Gemma/Gemma2 generator, on-premises embedder, versioned corpus with provenance, ablations over chunk size/overlap/k — confirms reduced hallucination on a single-GPU server with "predictable" costs and latency. **Receipt 2** (10.3390/bdcc9120320, *Big Data and Cognitive Computing*, 2025; openalex:full-corpus), a PRISMA 2020 systematic review, cautions that "empirical results are scattered across tasks, systems, and metrics, limiting cumulative insight," uses descriptive synthesis only, and flags the very ablations Receipt 1 highlights as not standardised. The supported/baselines/latency framing travels; the verdict does not.

## The 2+2=5 angle
The hidden boundary is *evaluation scope*. Receipt 1 is a single-domain, single-GPU, preregistered, openly released pipeline with internal ablations — a positive case study. Receipt 2 is a meta-survey spanning January 2020–May 2025 across ACM/IEEE/Scopus/ScienceDirect/DBLP with citation thresholds (≥15 for 2025; ≥30 for 2024 or earlier) — and explicitly declines meta-analysis. Reading both as "RAG works" or "RAG fails" is a metric mismatch: one is an artefact benchmark, the other is a corpus-level negative signal on comparability. The 2+2=5 emerges when investors price RAG-as-factuality on Receipt 1's "confirmed" framing without Receipt 2's "scattered, no meta-analysis" caveat on latency-vs-baselines tradeoffs.

## Why this could matter
For vendors selling RAG "factuality" to clinical/regulated buyers, the **supported** claims are bounded by what Receipt 1's ablations actually test; Receipt 2 shows those ablations are not yet standard, so comparative **baselines** and **latency** claims cannot be normalised across offerings. Decision support RAG (clinical, single-GPU, curated corpus) and literature-retrieval RAG (broad, parametric-plus-retrieval, multi-domain) sit on different cost/latency curves — a cross-domain transfer hypothesis, not a settled equivalence.

## What would break the idea
- A meta-analysis or shared latency-vs-faithfulness benchmark (hypothesis, since Receipt 2 reports none).
- Receipt 1's pipeline failing to reproduce beyond the single-GPU, curated-corpus setting.
- Receipt 2's descriptive counts shifting if 2025 citation-threshold adjustments distort coverage.

## Receipts
- 10.2339/politeknik.1810629 — openalex:full-corpus; trial/protocol-style pipeline, 2025.
- 10.3390/bdcc9120320 — openalex:full-corpus; market study/systematic review, 2025.

## Safety note
Hypothesis, not advice. Receipt 1's "experimental evaluation confirmed" is scoped to its own pipeline; Receipt 2's synthesis is descriptive and non-meta-analytic. No clinical or procurement guidance.

