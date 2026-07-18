from v5_memo.gate import candidate_publish_blocker
from v5_memo.miner import _claim_card
from v5_memo.schemas import ClaimCard, CorpusHit, InsightCandidate, ReceiptRole
from v5_memo.writer import _intervention_context, render_memo


def _candidate(*cards: ClaimCard) -> InsightCandidate:
    return InsightCandidate(
        topic="supplement muscle performance",
        thesis="The receipt-bound directions may differ across muscle endpoints.",
        bridge_terms=("supplement", "muscle"),
        tension_terms=("null", "positive"),
        receipt_ids=tuple(card.receipt_id for card in cards),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:boundary_condition", "tier:publishable_alpha"),
        claim_cards=cards,
    )


def test_human_trial_population_overrides_incidental_animal_background() -> None:
    hit = CorpusHit(
        hit_id="human-rct",
        title="Supplement effects on muscle performance",
        abstract=(
            "Benefits were previously reported in aged mice and other animals. "
            "In this randomized trial, 120 older adult participants received the supplement."
        ),
        source="fullraw:openalex",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "human trial"))

    assert card.design == "randomized_trial"
    assert card.population == "human"
    assert card.support_type == "direct"


def test_animal_study_remains_indirect_despite_human_background() -> None:
    hit = CorpusHit(
        hit_id="rat-study",
        title="Supplement response in randomized male Wistar rats",
        abstract="The introduction notes human relevance; 32 rats received the intervention.",
        source="fullraw:openalex",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "model study"))

    assert card.population == "animal"
    assert card.support_type == "indirect"


def test_current_animal_experiment_overrides_prior_human_trial_background() -> None:
    hit = CorpusHit(
        hit_id="current-mice",
        title="Supplement effects on muscle performance",
        abstract=(
            "In the current experiment, 40 mice were assigned to the supplement. "
            "In a prior study, human participants were randomized to treatment."
        ),
        source="fullraw:openalex",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "model study"))

    assert card.population == "animal"
    assert card.support_type == "indirect"


def test_previously_sedentary_participants_remain_current_human_population() -> None:
    hit = CorpusHit(
        hit_id="sedentary-human-rct",
        title="Supplement effects on muscle performance",
        abstract=(
            "Participants previously sedentary were randomized to the supplement. "
            "Mouse models motivated this trial."
        ),
        source="fullraw:openalex",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "human trial"))

    assert card.population == "human"
    assert card.support_type == "direct"


def test_female_randomized_trial_is_direct_human_evidence() -> None:
    hit = CorpusHit(
        hit_id="10.1210/jc.2012-2340",
        title=(
            "Skeletal Muscle Strength in Young Asian Indian Females after Vitamin D and Calcium "
            "Supplementation: A Double-Blind Randomized Controlled Clinical Trial"
        ),
        abstract=(
            "Oral cholecalciferol/calcium supplementation in the dose/schedule used is effective "
            "and safe in increasing and maintaining serum 25(OH)D. However, this does not lead to "
            "improved skeletal muscle strength in young females."
        ),
        source="fullraw:openalex",
        doi="10.1210/jc.2012-2340",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "null_signal", "human trial"))

    assert card.design == "randomized_trial"
    assert card.population == "human"
    assert card.direction == "null"
    assert card.support_type == "direct"
    assert card.confidence == "high"


def test_current_female_trial_overrides_prior_animal_background() -> None:
    hit = CorpusHit(
        hit_id="female-rct",
        title="Supplement effects on muscle strength",
        abstract=(
            "Previous mouse models motivated the study. "
            "Young females were randomized to the supplement and placebo."
        ),
        source="fullraw:openalex",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "human trial"))

    assert card.population == "human"
    assert card.support_type == "direct"


def test_publish_gate_rejects_unmapped_multi_direction_primary_card() -> None:
    mixed = ClaimCard(
        "strength",
        "positive_signal",
        "randomized_trial",
        "human",
        "muscle strength",
        "null/positive",
        "direct",
        "high",
        "The supplement improved strength, while another endpoint was unchanged.",
    )
    support = ClaimCard(
        "endurance",
        "positive_signal",
        "randomized_trial",
        "human",
        "muscle endurance",
        "positive",
        "direct",
        "high",
        "The supplement improved muscle endurance.",
    )

    assert candidate_publish_blocker(_candidate(mixed, support)) == {
        "error": "ambiguous_direction_without_endpoint_mapping",
        "receipt_ids": ("strength",),
    }


def test_publish_gate_accepts_explicit_endpoint_direction_mapping() -> None:
    mapped = ClaimCard(
        "strength",
        "positive_signal",
        "randomized_trial",
        "human",
        "muscle strength=positive/fatigue=null",
        "null/positive",
        "direct",
        "high",
        "The supplement improved muscle strength while fatigue was unchanged.",
    )
    support = ClaimCard(
        "endurance",
        "positive_signal",
        "randomized_trial",
        "human",
        "muscle endurance",
        "positive",
        "direct",
        "high",
        "The supplement improved muscle endurance.",
    )

    assert candidate_publish_blocker(_candidate(mapped, support)) is None


def test_claim_card_emits_mapping_only_for_separately_bound_endpoints() -> None:
    hit = CorpusHit(
        hit_id="mapped-rct",
        title="Supplement improved muscle strength but there was no difference in fatigue",
        abstract="Human participants were randomized and received the supplement.",
        source="fullraw:openalex",
    )
    mapped = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "human trial"))
    support = ClaimCard(
        "endurance",
        "positive_signal",
        "randomized_trial",
        "human",
        "muscle endurance",
        "positive",
        "direct",
        "high",
        "The supplement improved muscle endurance.",
    )

    assert mapped.outcome == "muscle strength=positive/fatigue=null"
    assert mapped.direction == "null/positive"
    assert candidate_publish_blocker(_candidate(mapped, support)) is None


def test_claim_card_binds_nominal_improvement_to_endpoint_not_reporting_verb() -> None:
    hit = CorpusHit(
        hit_id="nominal-rct",
        title="The trial found improvement in muscle strength but no difference in fatigue",
        abstract="Human participants were randomized and received the supplement.",
        source="fullraw:openalex",
    )
    mapped = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "human trial"))
    support = ClaimCard(
        "endurance",
        "positive_signal",
        "randomized_trial",
        "human",
        "muscle endurance",
        "positive",
        "direct",
        "high",
        "The supplement improved muscle endurance.",
    )

    assert mapped.outcome == "muscle strength=positive/fatigue=null"
    assert "found=" not in mapped.outcome
    assert candidate_publish_blocker(_candidate(mapped, support)) is None


def test_claim_card_binds_related_nominal_direction_variants() -> None:
    for noun, direction in (("increase", "positive"), ("reduction", "negative"), ("decrease", "negative")):
        hit = CorpusHit(
            hit_id=f"nominal-{noun}",
            title=f"The trial observed a {noun} in muscle strength but no difference in fatigue",
            abstract="Human participants were randomized and received the supplement.",
            source="fullraw:openalex",
        )

        card = _claim_card(hit, ReceiptRole(hit.hit_id, f"{direction}_signal", "human trial"))

        assert card.outcome == f"muscle strength={direction}/fatigue=null"


def test_claim_card_fails_closed_for_unknown_nominal_endpoint() -> None:
    hit = CorpusHit(
        hit_id="unknown-endpoint",
        title="The trial reported improvement in zorbularity but no difference in fatigue",
        abstract="Human participants were randomized and received the supplement.",
        source="fullraw:openalex",
    )
    mixed = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "human trial"))
    support = ClaimCard(
        "endurance",
        "positive_signal",
        "randomized_trial",
        "human",
        "muscle endurance",
        "positive",
        "direct",
        "high",
        "The supplement improved muscle endurance.",
    )

    assert "=" not in mixed.outcome
    assert candidate_publish_blocker(_candidate(mixed, support)) == {
        "error": "ambiguous_direction_without_endpoint_mapping",
        "receipt_ids": ("unknown-endpoint",),
    }


def test_claim_card_leaves_unbound_mixed_direction_blocked() -> None:
    hit = CorpusHit(
        hit_id="ambiguous-rct",
        title="Supplement improved muscle performance and was unchanged overall",
        abstract="Human participants were randomized and received the supplement.",
        source="fullraw:openalex",
    )
    mixed = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "human trial"))
    support = ClaimCard(
        "endurance",
        "positive_signal",
        "randomized_trial",
        "human",
        "muscle endurance",
        "positive",
        "direct",
        "high",
        "The supplement improved muscle endurance.",
    )

    assert "=" not in mixed.outcome
    assert candidate_publish_blocker(_candidate(mixed, support)) == {
        "error": "ambiguous_direction_without_endpoint_mapping",
        "receipt_ids": ("ambiguous-rct",),
    }


def test_receipt_synthesis_integrates_population_endpoint_dose_and_duration() -> None:
    cards = (
        ClaimCard(
            "strength",
            "positive_signal",
            "randomized_trial",
            "human",
            "muscle strength",
            "positive",
            "direct",
            "high",
            "The supplement improved muscle strength.",
        ),
        ClaimCard(
            "endurance",
            "positive_signal",
            "randomized_trial",
            "human",
            "muscle endurance",
            "positive",
            "direct",
            "high",
            "The supplement improved muscle endurance.",
        ),
    )
    receipts = (
        CorpusHit(
            hit_id="strength",
            title="Supplement improves muscle strength in adults",
            abstract="Participants received 500 mg daily for 12 weeks.",
            source="fullraw:openalex",
        ),
        CorpusHit(
            hit_id="endurance",
            title="Supplement improves muscle endurance in adults",
            abstract="Participants received 1 g per day over 6 months.",
            source="fullraw:openalex",
        ),
    )

    memo = render_memo(_candidate(*cards), receipts)

    assert "randomized trial in human; endpoint: muscle strength; direction: improved; dose: 500 mg daily; duration: for 12 weeks" in memo
    assert "randomized trial in human; endpoint: muscle endurance; direction: improved; dose: 1 g per day; duration: over 6 months" in memo


def test_intervention_context_does_not_mix_prior_and_current_regimens() -> None:
    hit = CorpusHit(
        hit_id="current-regimen",
        title="Supplement trial in older adults",
        abstract=(
            "Prior mice received 500 mg daily for 12 weeks. "
            "In the current human trial, participants received 250 mg daily for 8 weeks."
        ),
        source="fullraw:openalex",
    )

    assert _intervention_context(hit) == ("250 mg daily", "for 8 weeks")


def test_intervention_context_preserves_comma_grouped_dose() -> None:
    hit = CorpusHit(
        hit_id="comma-dose",
        title="Supplement trial in older adults",
        abstract="Participants received 1,000 mg daily for 4 weeks.",
        source="fullraw:openalex",
    )

    assert _intervention_context(hit) == ("1,000 mg daily", "for 4 weeks")


def test_intervention_context_rejects_ambiguous_multi_arm_regimen() -> None:
    hit = CorpusHit(
        hit_id="multi-arm",
        title="Two-dose supplement trial",
        abstract="Participants received either 250 mg or 500 mg daily for 8 weeks.",
        source="fullraw:openalex",
    )

    assert _intervention_context(hit) == ("not stated", "not stated")
