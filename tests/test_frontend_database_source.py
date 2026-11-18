from pathlib import Path

from database import database
from database.database import init_db
from services import repository as repo
from services.research_results_service import (
    get_sensitivity_results,
    get_poc_results,
    get_tournament_results,
    get_champion_results,
)


def _tmp_db(monkeypatch, tmp_path):
    db = tmp_path / 'frontend_source.db'
    monkeypatch.setattr(database, 'DB_PATH', db)
    init_db()
    return db


def test_simple_frontend_does_not_import_reference_contract():
    root = Path(__file__).resolve().parents[1] / 'ui'
    for name in ['simple_dashboard.py','simple_results.py','simple_prediction.py','simple_history.py']:
        text = (root / name).read_text(encoding='utf-8')
        assert 'reference_contract' not in text


def test_results_can_be_read_from_parity_report_database(monkeypatch, tmp_path):
    _tmp_db(monkeypatch, tmp_path)
    did = repo.create_dataset('Research', 'BTC-USD', '2021-01-01', '2026-04-01', '1d', is_locked=True)
    report = {
        'sections': {
            'sensitivity': {'actual': [
                {'n_hidden': 10, 'C': 100.0, 'MAE Mean': 1.0, 'RMSE Mean': 2.0, 'MAPE Mean': 3.0, 'MAPE Std': 0.1}
            ]},
            'poc': {'summary': {
                'elm_mae': 1.0, 'elm_rmse': 2.0, 'elm_mape': 3.0, 'elm_std': 0.3,
                'pso_mae': 0.8, 'pso_rmse': 1.5, 'pso_mape': 2.0, 'pso_std': 0.2,
                'delta_mape': 33.33, 'p_value': 0.001, 'cohens_d': 1.2,
            }},
            'tournament': {'table': [
                {'Skenario': 'S4 (Tren SMA)', 'ELM MAPE': 3.0, 'ELM Std': 0.2, 'PSO MAPE': 2.0, 'PSO Std': 0.1, 'ΔMAPE': 33.33, 'p-value': 0.001}
            ]},
            'champion': {
                'summary': {'mae': 100.0, 'rmse': 150.0, 'mape': 2.0},
                'folds': [{'fold': 1, 'train_size': 10, 'test_size': 5, 'n_hidden': 12, 'C': 1000.0, 'mae': 100.0, 'rmse': 150.0, 'mape': 2.0}],
            },
        },
        'passed': False,
        'aborted': False,
    }
    repo.save_parity_report(did, 'TEST_REFERENCE', 'full', report)

    sens, _ = get_sensitivity_results()
    poc, _ = get_poc_results()
    tour, _ = get_tournament_results()
    champ, folds, _ = get_champion_results()

    assert float(sens.iloc[0]['MAPE Mean']) == 3.0
    assert poc['pso_mape'] == 2.0
    assert float(tour.iloc[0]['PSO MAPE']) == 2.0
    assert champ['mape'] == 2.0
    assert int(folds.iloc[0]['fold_number']) == 1
    assert float(folds.iloc[0]['c_value']) == 1000.0


def test_sensitivity_run_hyperparameters_are_persisted(monkeypatch, tmp_path):
    _tmp_db(monkeypatch, tmp_path)
    did = repo.create_dataset('Research', 'BTC-USD', '2021-01-01', '2026-04-01', '1d')
    eid = repo.create_experiment('Sensitivity', 'sensitivity', did)
    repo.save_runs(eid, [{
        'scenario': 'S3 (Momentum - Teori)', 'model_type': 'ELM', 'run_number': 1, 'seed': 0,
        'n_hidden': 10, 'C': 100.0, 'mae': 1.0, 'rmse': 2.0, 'mape': 3.0,
    }])
    repo.finish_experiment(eid)
    rows = repo.experiment_runs(eid)
    assert rows[0]['n_hidden'] == 10
    assert rows[0]['c_value'] == 100.0
