import pytest

from agents.tools.merit import (
    calculate_aggregate,
    comsats_merit,
    fast_merit,
    giki_merit,
    nust_merit,
    uet_merit,
)


def test_calculate_aggregate_basic():
    marks = {"a": 100, "b": 50}
    weights = {"a": 0.5, "b": 0.5}
    assert calculate_aggregate(marks, weights) == pytest.approx(75.0)


def test_fast_merit_bs():
    # 50% test + 40% fsc + 10% matric
    result = fast_merit(matric_pct=90, fsc_pct=85, test_pct=80, program="bs")
    expected = 0.50 * 80 + 0.40 * 85 + 0.10 * 90
    assert result == pytest.approx(expected)


def test_fast_merit_engineering():
    # 33% test + 50% fsc + 17% matric
    result = fast_merit(matric_pct=90, fsc_pct=85, test_pct=80, program="engineering")
    expected = 0.33 * 80 + 0.50 * 85 + 0.17 * 90
    assert result == pytest.approx(expected)


def test_comsats_merit():
    # 10% matric + 40% fsc + 50% nts
    result = comsats_merit(matric_pct=90, fsc_pct=85, nts_pct=70)
    expected = 0.10 * 90 + 0.40 * 85 + 0.50 * 70
    assert result == pytest.approx(expected)


def test_giki_merit():
    # 85% test + 15% ssc
    result = giki_merit(ssc_pct=90, test_pct=75)
    expected = 0.85 * 75 + 0.15 * 90
    assert result == pytest.approx(expected)


def test_uet_merit():
    # 33% ecat + 50% fsc + 17% matric
    result = uet_merit(matric_pct=90, fsc_pct=85, ecat_pct=70)
    expected = 0.33 * 70 + 0.50 * 85 + 0.17 * 90
    assert result == pytest.approx(expected)


def test_nust_merit():
    # 75% net + 25% average(matric, hssc)
    result = nust_merit(matric_pct=90, hssc_pct=80, net_pct=70)
    expected = 0.75 * 70 + 0.25 * ((90 + 80) / 2)
    assert result == pytest.approx(expected)


def test_perfect_scores_give_100():
    assert fast_merit(matric_pct=100, fsc_pct=100, test_pct=100) == pytest.approx(100.0)
    assert comsats_merit(matric_pct=100, fsc_pct=100, nts_pct=100) == pytest.approx(100.0)
    assert giki_merit(ssc_pct=100, test_pct=100) == pytest.approx(100.0)
    assert uet_merit(matric_pct=100, fsc_pct=100, ecat_pct=100) == pytest.approx(100.0)
    assert nust_merit(matric_pct=100, hssc_pct=100, net_pct=100) == pytest.approx(100.0)
