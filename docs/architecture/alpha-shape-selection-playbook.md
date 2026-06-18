# Alpha Shape Selection Playbook

V5 should improve business and AI memos by selecting sharper receipt pairs before writing, not by hardcoding domains or adding prose bureaucracy.

## Universal Ranking Rule

Prefer receipt pairs with one of these shapes:

1. Same construct, opposite direction.
2. Same intervention, tool, or policy, different endpoint.
3. Stated intent, theory, protocol, or managerial claim versus observed result.
4. Benefit at one layer, cost at another layer.
5. Specific boundary condition that changes the interpretation of both receipts.

Demote pairs with these weaker shapes:

1. One local success plus one broad review saying evidence is mixed.
2. Two papers sharing keywords but not the same construct.
3. Generic "more research is needed" caveats.
4. Claims where the interesting part is not directly supported by the receipts.

## Example Search Angles: Business

These are query-seeding examples only. Runtime selection must not require
business-specific terms; it should rank the universal alpha shape after receipts
are mined and bound.

Business alpha should look for decisions where the intended control channel and the observed market or organizational response diverge.

Useful shapes:

- Disclosure improves transparency but changes behavior in the wrong direction.
- Incentives align one metric while creating gaming, misreporting, or effort substitution.
- Forecasting tools improve precision but reduce judgment, coverage, or accountability.
- More management information worsens one forecast metric while making analyst or market signals more valuable.
- Regulation reduces one risk while shifting the risk to timing, classification, opacity, or intermediaries.

Search-angle terms:

- `guidance`, `forecast`, `analyst revision`, `forecast dispersion`, `forecast error`
- `disclosure`, `opacity`, `transparency`, `information environment`
- `incentive`, `bonus`, `quota`, `KPI`, `gaming`, `misreporting`
- `restatement`, `audit`, `compliance`, `regulation`, `risk disclosure`
- `managerial attention`, `decision quality`, `automation`, `dashboard`

Good memo pattern:

> The policy/tool appears to improve one information channel, but the receipts show value moving to a different channel or cost surfacing in a different metric.

## Example Search Angles: AI

These are query-seeding examples only. Runtime selection must not require
AI-specific terms; it should rank the universal alpha shape after receipts are
mined and bound.

AI alpha should avoid generic benchmark-heterogeneity memos unless the boundary is unusually specific and receipt-bound.

Useful shapes:

- Benchmark score improves while calibration, robustness, or tail reliability worsens.
- A safety method reduces one failure mode but increases abstention, refusal, latency, or hidden error.
- Retrieval or longer context improves factuality on one task but harms reasoning, speed, or comparability elsewhere.
- Human-AI assistance raises average productivity while degrading review quality, learning, or error detection.
- Scaling improves average performance while worsening overconfidence, contamination exposure, or distribution-shift behavior.

Search-angle terms:

- `benchmark contamination`, `leaderboard`, `evaluation leakage`, `memorization`
- `calibration`, `overconfidence`, `selective prediction`, `abstention`
- `faithfulness`, `factuality`, `hallucination`, `retrieval augmented generation`
- `automation bias`, `human oversight`, `deskilling`, `review quality`
- `distribution shift`, `robustness`, `tail risk`, `latency`, `context length`

Good memo pattern:

> The model or method improves the headline metric, but the receipts show a hidden reliability, comparability, or human-system cost.

## Implementation Implication

Keep this as a selector preference, not a domain-specific rule:

- Generate diverse search queries.
- Mine and bind receipt pairs.
- Rank candidate pairs by alpha shape before MiniMax writes, using domain-neutral
  shape signals such as opposite direction, boundary condition, intent versus
  observed result, and benefit/cost tradeoff.
- Let MiniMax write only after the receipt pair already has a strong shape.

The writer should polish the insight. It should not rescue a weak pair.
