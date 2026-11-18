import pandas as pd
from research_engine.parity import fold_plan, _usd_eq, USD_ABS_TOL
from research_engine.reference_contract import WFV_REFERENCE, DATA_REFERENCE
from research_engine.config import ELM_BASELINE_NH, ELM_BASELINE_C, TOURNAMENT_PSO_SEED


def test_reference_fold_plan():
    assert fold_plan(DATA_REFERENCE['processed_samples']) == WFV_REFERENCE


def test_final_baseline_contract():
    assert ELM_BASELINE_NH == 10
    assert ELM_BASELINE_C == 1e2


def test_final_tournament_seed_contract():
    assert [TOURNAMENT_PSO_SEED(i) for i in range(3)] == [42, 55, 68]


def test_usd_tolerance_accepts_one_cent_blas_variance():
    assert USD_ABS_TOL == 0.02
    assert _usd_eq(2695.84, 2695.83)
    assert _usd_eq(3688.34, 3688.33)
    assert not _usd_eq(2695.86, 2695.83)
