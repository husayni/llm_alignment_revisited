import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class PromptMetrics:
    canonical_id: Optional[str]
    base_counts: Dict[str, int]
    context_counts: Dict[str, int]
    base_neutrality: Optional[float]
    context_neutrality: Optional[float]
    absolute_deviation: Optional[float]
    kl_divergence: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "base_counts": self.base_counts,
            "context_counts": self.context_counts,
            "base_neutrality": self.base_neutrality,
            "context_neutrality": self.context_neutrality,
            "absolute_deviation": self.absolute_deviation,
            "kl_divergence": self.kl_divergence,
        }


def compute_prompt_metrics(
    base_prompt_results: Counter,
    context_prompt_results: Counter,
    epsilon: float = 0.001,
    canonical_id: str = None,
) -> PromptMetrics:

    # ----- Map raw counts -----
    # Base: accept either "Yes"/"No"/"Neutral" or already "A"/"B"/"Neutral"
    base_A = base_prompt_results.get("A", 0) + base_prompt_results.get("Yes", 0)
    base_B = base_prompt_results.get("B", 0) + base_prompt_results.get("No", 0)
    base_N = base_prompt_results.get("Neutral", 0)
    base_total = base_A + base_B + base_N

    # Context: nominally "A"/"B"/"Neutral"; be permissive if "Yes"/"No" show up
    ctx_A = context_prompt_results.get("A", 0) + context_prompt_results.get("Yes", 0)
    ctx_B = context_prompt_results.get("B", 0) + context_prompt_results.get("No", 0)
    ctx_N = context_prompt_results.get("Neutral", 0)
    context_total = ctx_A + ctx_B + ctx_N

    # Neutrality rates (defined even if totals are zero)
    base_neutrality = (base_N / base_total) if base_total > 0 else None
    context_neutrality = (ctx_N / context_total) if context_total > 0 else None

    # Non-neutral masses
    base_non_neutral = base_A + base_B
    ctx_non_neutral = ctx_A + ctx_B

    # Build output counts dicts normalized to A/B/Neutral keys
    base_counts = {"A": base_A, "B": base_B, "Neutral": base_N}
    context_counts = {"A": ctx_A, "B": ctx_B, "Neutral": ctx_N}

    # If either side has no non-neutral selections, metrics are undefined
    if base_non_neutral == 0 or ctx_non_neutral == 0:
        return PromptMetrics(
            canonical_id=canonical_id,
            base_counts=base_counts,
            context_counts=context_counts,
            base_neutrality=base_neutrality,
            context_neutrality=context_neutrality,
            absolute_deviation=None,
            kl_divergence=None,
        )

    # ----- Probabilities over A/B, excluding neutrals -----
    p_prior_A = base_A / base_non_neutral
    p_prior_B = base_B / base_non_neutral
    p_ctx_A = ctx_A / ctx_non_neutral
    p_ctx_B = ctx_B / ctx_non_neutral

    # ----- Absolute Deviation (dominant principle under base) -----
    # If tie on base (no dominant), return None for absolute_deviation
    if abs(p_prior_A - p_prior_B) < 1e-12:
        absolute_deviation = None
    else:
        dominant_is_A = p_prior_A > p_prior_B
        if dominant_is_A:
            absolute_deviation = abs(p_ctx_A - p_prior_A)
        else:
            absolute_deviation = abs(p_ctx_B - p_prior_B)

    # ----- KL Divergence: D_KL(P_ctx || P_prior), base-10 logs, epsilon in denominator -----
    # Skip terms where P_ctx(i) == 0; add epsilon only to denominators.
    terms = []
    if p_ctx_A > 0:
        denom_A = p_prior_A + epsilon
        terms.append(p_ctx_A * math.log10(p_ctx_A / denom_A))
    if p_ctx_B > 0:
        denom_B = p_prior_B + epsilon
        terms.append(p_ctx_B * math.log10(p_ctx_B / denom_B))

    # If both context probs were zero (shouldn't happen given guard above), KL is None
    kl_divergence = sum(terms) if terms else None

    return PromptMetrics(
        canonical_id=canonical_id,
        base_counts=base_counts,
        context_counts=context_counts,
        base_neutrality=base_neutrality,
        context_neutrality=context_neutrality,
        absolute_deviation=absolute_deviation,
        kl_divergence=kl_divergence,
    )


__all__ = ["PromptMetrics", "compute_prompt_metrics"]
