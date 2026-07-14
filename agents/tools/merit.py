"""
Deterministic merit-aggregate calculators, one per university — pure Python,
no LLM involved. This is the project's core anti-hallucination moat: the
ARITHMETIC must always come from code, never from an LLM computing a weighted
sum in its head (a well-known LLM failure mode even given correct inputs).

The WEIGHTS below are data, not permanent facts — they change every admission
cycle. When the corpus is refreshed for a new cycle, update MERIT_FORMULAS
here (verified by hand against the new prospectus, same way we extracted
these originally — see corpus/sources.md). The calculation logic itself
never needs to change when weights change.
"""

MERIT_FORMULAS = {
    "fast": {
        "bs": {"test": 0.50, "fsc": 0.40, "matric": 0.10},
        "engineering": {"test": 0.33, "fsc": 0.50, "matric": 0.17},  # per PEC recommendation
    },
    "comsats": {
        # same formula for Engineering and non-Engineering; NTS test requires
        # a minimum of 33% per PEC policy — that's an eligibility check,
        # handled separately from this aggregate calculation.
        "default": {"nts": 0.50, "fsc": 0.40, "matric": 0.10},
    },
    "giki": {
        "default": {"test": 0.85, "ssc": 0.15},
    },
    "uet": {
        # FSc-stream only for v1; O/A-level and diploma variants are a
        # documented simplification (see corpus/sources.md / PRD non-goals).
        "default": {"ecat": 0.33, "fsc": 0.50, "matric": 0.17},
    },
    "nust": {
        # simplified for v1 — real formula has subject-specific NET
        # weightings per program; see corpus/sources.md for the full detail.
        "default": {"net": 0.75, "academic": 0.25},
    },
}


def calculate_aggregate(marks: dict[str, float], weights: dict[str, float]) -> float:
    """
    Generic weighted-sum aggregate calculator, reused by every university.
    marks and weights must share the same keys (e.g. {"test": ..., "fsc": ...}).
    """
    return sum(marks[key] * weight for key, weight in weights.items())


def fast_merit(matric_pct: float, fsc_pct: float, test_pct: float, program: str = "bs") -> float:
    weights = MERIT_FORMULAS["fast"]["engineering" if program == "engineering" else "bs"]
    marks = {"test": test_pct, "fsc": fsc_pct, "matric": matric_pct}
    return calculate_aggregate(marks, weights)


def comsats_merit(matric_pct: float, fsc_pct: float, nts_pct: float) -> float:
    weights = MERIT_FORMULAS["comsats"]["default"]
    marks = {"nts": nts_pct, "fsc": fsc_pct, "matric": matric_pct}
    return calculate_aggregate(marks, weights)


def giki_merit(ssc_pct: float, test_pct: float) -> float:
    weights = MERIT_FORMULAS["giki"]["default"]
    marks = {"test": test_pct, "ssc": ssc_pct}
    return calculate_aggregate(marks, weights)


def uet_merit(matric_pct: float, fsc_pct: float, ecat_pct: float) -> float:
    weights = MERIT_FORMULAS["uet"]["default"]
    marks = {"ecat": ecat_pct, "fsc": fsc_pct, "matric": matric_pct}
    return calculate_aggregate(marks, weights)


def nust_merit(matric_pct: float, hssc_pct: float, net_pct: float) -> float:
    weights = MERIT_FORMULAS["nust"]["default"]
    # v1 simplification: matric + hssc averaged into a single "academic" mark
    marks = {"net": net_pct, "academic": (matric_pct + hssc_pct) / 2}
    return calculate_aggregate(marks, weights)
