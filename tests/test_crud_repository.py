import pandas as pd
import pytest

from database import database
from database.database import init_db
from services import repository as repo


def _tmp_db(monkeypatch, tmp_path):
    db = tmp_path / 'test.db'
    monkeypatch.setattr(database, 'DB_PATH', db)
    init_db()
    return db


def test_dataset_crud_and_locked_guard(monkeypatch, tmp_path):
    _tmp_db(monkeypatch, tmp_path)
    did = repo.create_dataset('Custom Test', 'BTC-USD', '2026-01-01', '2026-01-04', '1d', mode='custom')
    assert repo.get_dataset(did)['name'] == 'Custom Test'

    idx = pd.date_range('2026-01-01', periods=3, freq='D')
    df = pd.DataFrame({
        'Open':[1,2,3], 'High':[2,3,4], 'Low':[0,1,2], 'Close':[1.5,2.5,3.5], 'Volume':[10,11,12]
    }, index=idx)
    repo.save_market_data(did, df)
    assert len(repo.load_market_data(did)) == 3

    repo.update_dataset(did, 'Custom Updated', 'updated')
    row = repo.get_dataset(did)
    assert row['name'] == 'Custom Updated' and row['description'] == 'updated'
    assert repo.delete_dataset(did) is True
    assert repo.get_dataset(did) is None

    locked = repo.create_dataset('Locked Research', 'BTC-USD', '2021-01-01', '2026-04-01', '1d', is_locked=True)
    with pytest.raises(ValueError):
        repo.delete_dataset(locked)
    assert repo.get_dataset(locked) is not None


def test_experiment_crud(monkeypatch, tmp_path):
    _tmp_db(monkeypatch, tmp_path)
    did = repo.create_dataset('Research', 'BTC-USD', '2021-01-01', '2026-04-01', '1d')
    eid = repo.create_experiment('Experiment Test', 'poc', did, notes='awal', config={'x': 1})
    assert repo.get_experiment(eid)['status'] == 'running'
    repo.update_experiment(eid, 'Experiment Updated', 'catatan baru')
    row = repo.get_experiment(eid)
    assert row['name'] == 'Experiment Updated' and row['notes'] == 'catatan baru'
    repo.finish_experiment(eid)
    assert repo.get_experiment(eid)['status'] == 'completed'
    repo.delete_experiment(eid)
    assert repo.get_experiment(eid) is None
