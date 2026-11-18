import json, uuid
from pathlib import Path
import pandas as pd
from database.database import connect


def _id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def log(action, object_type=None, object_id=None, description=None):
    with connect() as con:
        con.execute(
            "INSERT INTO activity_logs(action,object_type,object_id,description) VALUES(?,?,?,?)",
            (action, object_type, object_id, description),
        )


# ----------------------------- DATASET --------------------------------------
def create_dataset(name, symbol, start, end, interval, description='', is_locked=False,
                   mode='research', source='Yahoo Finance'):
    did = _id('DS')
    with connect() as con:
        con.execute(
            "INSERT INTO datasets(id,name,symbol,start_date,end_date,interval,description,is_locked,mode,source) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (did, name, symbol, start, end, interval, description, int(is_locked), mode, source),
        )
    log('create', 'dataset', did, f'{name} • {mode}')
    return did


def list_datasets(mode=None):
    with connect() as con:
        if mode:
            rows = con.execute("SELECT * FROM datasets WHERE mode=? ORDER BY created_at DESC", (mode,)).fetchall()
        else:
            rows = con.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_dataset(dataset_id):
    with connect() as con:
        r = con.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return dict(r) if r else None


def latest_dataset_by_mode(mode):
    with connect() as con:
        r = con.execute(
            "SELECT * FROM datasets WHERE mode=? ORDER BY COALESCE(last_synced_at,created_at) DESC LIMIT 1", (mode,)
        ).fetchone()
        return dict(r) if r else None


def update_dataset(dataset_id, name, description):
    with connect() as con:
        con.execute(
            "UPDATE datasets SET name=?,description=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (name, description, dataset_id),
        )
    log('update', 'dataset', dataset_id, name)


def update_dataset_sync(dataset_id, end_date=None):
    with connect() as con:
        if end_date:
            con.execute(
                "UPDATE datasets SET end_date=?,last_synced_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (end_date, dataset_id),
            )
        else:
            con.execute(
                "UPDATE datasets SET last_synced_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (dataset_id,),
            )


def delete_dataset(dataset_id):
    with connect() as con:
        row = con.execute("SELECT is_locked FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        if not row:
            return False
        if int(row['is_locked'] or 0) == 1:
            raise ValueError('Dataset locked tidak dapat dihapus.')
        # Models may originate from a migrated schema where dataset_id has no FK.
        # Detach them first so deleting an unlocked dataset never leaves misleading references.
        con.execute("UPDATE models SET dataset_id=NULL WHERE dataset_id=?", (dataset_id,))
        con.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
    log('delete', 'dataset', dataset_id)
    return True


def save_market_data(dataset_id, df):
    rows = []
    for idx, row in df.iterrows():
        rows.append((
            dataset_id, pd.Timestamp(idx).strftime('%Y-%m-%d'), float(row['Open']),
            float(row['High']), float(row['Low']), float(row['Close']), float(row['Volume'])
        ))
    with connect() as con:
        con.executemany(
            "INSERT OR REPLACE INTO market_data(dataset_id,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)",
            rows,
        )
        con.execute("UPDATE datasets SET last_synced_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP WHERE id=?", (dataset_id,))
    log('snapshot', 'dataset', dataset_id, f'{len(rows)} market rows')


def load_market_data(dataset_id):
    with connect() as con:
        df = pd.read_sql_query(
            "SELECT date,open AS Open,high AS High,low AS Low,close AS Close,volume AS Volume "
            "FROM market_data WHERE dataset_id=? ORDER BY date",
            con, params=(dataset_id,),
        )
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.index.name = 'Date'
    return df


def dataset_market_stats(dataset_id):
    with connect() as con:
        r = con.execute(
            "SELECT COUNT(*) rows_count, MIN(date) first_date, MAX(date) last_date, "
            "(SELECT close FROM market_data WHERE dataset_id=? ORDER BY date DESC LIMIT 1) last_close "
            "FROM market_data WHERE dataset_id=?", (dataset_id, dataset_id)
        ).fetchone()
    return dict(r) if r else {'rows_count': 0, 'first_date': None, 'last_date': None, 'last_close': None}


# ---------------------------- EXPERIMENT ------------------------------------
def create_experiment(name, typ, dataset_id, notes='', config=None):
    eid = _id('EXP')
    with connect() as con:
        con.execute(
            "INSERT INTO experiments(id,name,experiment_type,dataset_id,status,notes,config_json,started_at) "
            "VALUES(?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
            (eid, name, typ, dataset_id, 'running', notes, json.dumps(config or {})),
        )
    log('create', 'experiment', eid, name)
    return eid


def finish_experiment(eid, status='completed'):
    with connect() as con:
        con.execute("UPDATE experiments SET status=?,completed_at=CURRENT_TIMESTAMP WHERE id=?", (status, eid))


def list_experiments(limit=100):
    with connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT e.*,d.name dataset_name,d.mode dataset_mode FROM experiments e "
            "JOIN datasets d ON d.id=e.dataset_id ORDER BY e.created_at DESC LIMIT ?", (limit,)
        ).fetchall()]


def get_experiment(eid):
    with connect() as con:
        r = con.execute(
            "SELECT e.*,d.name dataset_name,d.mode dataset_mode FROM experiments e JOIN datasets d ON d.id=e.dataset_id WHERE e.id=?",
            (eid,),
        ).fetchone()
        return dict(r) if r else None


def update_experiment(eid, name, notes):
    with connect() as con:
        con.execute("UPDATE experiments SET name=?,notes=? WHERE id=?", (name, notes, eid))
    log('update', 'experiment', eid, name)


def delete_experiment(eid):
    with connect() as con:
        con.execute("DELETE FROM experiments WHERE id=?", (eid,))
    log('delete', 'experiment', eid)


def save_runs(eid, runs):
    with connect() as con:
        for r in runs:
            cur = con.execute(
                "INSERT INTO experiment_runs(experiment_id,scenario,model_type,run_number,seed,mae,rmse,mape,execution_time,n_hidden,c_value) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (eid, r.get('scenario'), r.get('model_type'), r.get('run_number'), r.get('seed'),
                 r.get('mae'), r.get('rmse'), r.get('mape'), r.get('execution_time'),
                 r.get('n_hidden'), r.get('C')),
            )
            rid = cur.lastrowid
            for f in r.get('folds', []):
                con.execute(
                    "INSERT INTO fold_results(experiment_run_id,fold_number,train_size,test_size,n_hidden,c_value,mae,rmse,mape) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (rid, f.get('fold'), f.get('train_size'), f.get('test_size'), f.get('n_hidden'),
                     f.get('C'), f.get('mae'), f.get('rmse'), f.get('mape')),
                )


def experiment_runs(eid):
    with connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM experiment_runs WHERE experiment_id=? ORDER BY scenario,model_type,run_number", (eid,)
        ).fetchall()]


def experiment_folds(eid):
    with connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT f.*,r.scenario,r.model_type,r.run_number,r.seed FROM fold_results f "
            "JOIN experiment_runs r ON r.id=f.experiment_run_id WHERE r.experiment_id=? "
            "ORDER BY r.scenario,r.model_type,r.run_number,f.fold_number", (eid,)
        ).fetchall()]


def latest_scenario_mape(scenario, model_type='PSO-ELM'):
    with connect() as con:
        r = con.execute(
            "SELECT AVG(r.mape) mean_mape, e.id experiment_id FROM experiment_runs r "
            "JOIN experiments e ON e.id=r.experiment_id "
            "WHERE r.scenario=? AND r.model_type=? AND e.status='completed' "
            "GROUP BY e.id ORDER BY e.completed_at DESC LIMIT 1",
            (scenario, model_type),
        ).fetchone()
        return dict(r) if r else None


# ------------------------------- MODEL --------------------------------------
def create_model(dataset_id, scenario, algorithm, n_hidden, c_value, feature_set, model_path,
                 mape=None, experiment_id=None, trained_until=None, status='active', config=None, notes='',
                 model_mode='research', source_latest_date=None):
    mid = _id('MDL')
    with connect() as con:
        if status == 'active':
            con.execute("UPDATE models SET status='inactive' WHERE status='active' AND model_mode=?", (model_mode,))
        con.execute(
            "INSERT INTO models(id,experiment_id,dataset_id,scenario,algorithm,n_hidden,c_value,feature_set,mape,model_path,status,trained_until,config_json,notes,model_mode,source_latest_date) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, experiment_id, dataset_id, scenario, algorithm, n_hidden, c_value,
             json.dumps(feature_set), mape, str(model_path), status, trained_until,
             json.dumps(config or {}), notes, model_mode, source_latest_date),
        )
    log('create', 'model', mid, f'{algorithm} {scenario} • {model_mode}')
    return mid


def list_models(mode=None):
    with connect() as con:
        if mode:
            rows = con.execute(
                "SELECT m.*,d.name dataset_name FROM models m LEFT JOIN datasets d ON d.id=m.dataset_id "
                "WHERE m.model_mode=? ORDER BY m.created_at DESC", (mode,)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT m.*,d.name dataset_name FROM models m LEFT JOIN datasets d ON d.id=m.dataset_id ORDER BY m.created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]


def get_model(model_id):
    with connect() as con:
        r = con.execute(
            "SELECT m.*,d.name dataset_name FROM models m LEFT JOIN datasets d ON d.id=m.dataset_id WHERE m.id=?",
            (model_id,),
        ).fetchone()
        return dict(r) if r else None


def set_active_model(model_id):
    row = get_model(model_id)
    if not row:
        return
    mode = row.get('model_mode') or 'research'
    with connect() as con:
        con.execute("UPDATE models SET status='inactive' WHERE model_mode=?", (mode,))
        con.execute("UPDATE models SET status='active' WHERE id=?", (model_id,))
    log('activate', 'model', model_id, mode)


def delete_model(model_id):
    row = get_model(model_id)
    with connect() as con:
        con.execute("DELETE FROM models WHERE id=?", (model_id,))
    if row and row.get('model_path'):
        p = Path(row['model_path'])
        if p.exists():
            try: p.unlink()
            except OSError: pass
    log('delete', 'model', model_id)


def active_model(mode='research'):
    with connect() as con:
        r = con.execute(
            "SELECT m.*,d.name dataset_name FROM models m LEFT JOIN datasets d ON d.id=m.dataset_id "
            "WHERE m.status='active' AND m.model_mode=? ORDER BY m.created_at DESC LIMIT 1", (mode,)
        ).fetchone()
        return dict(r) if r else None


# ----------------------------- PREDICTION -----------------------------------
def create_prediction(model_id, input_date, prediction_date, last_close, predicted_close,
                      delta_usd, delta_pct, feature_snapshot=None, prediction_mode='research',
                      data_freshness_date=None, source_type='operational'):
    pid = _id('PRD')
    with connect() as con:
        con.execute(
            "INSERT INTO predictions(id,model_id,input_date,prediction_date,last_close,predicted_close,delta_usd,delta_pct,feature_json,prediction_mode,data_freshness_date,source_type) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, model_id, input_date, prediction_date, last_close, predicted_close,
             delta_usd, delta_pct, json.dumps(feature_snapshot or {}), prediction_mode,
             data_freshness_date or input_date, source_type),
        )
    log('create', 'prediction', pid, f'{prediction_mode}/{source_type} H+1 {prediction_date}')
    return pid


def list_predictions(limit=100, mode=None):
    with connect() as con:
        if mode:
            rows = con.execute(
                "SELECT p.*,m.scenario,m.algorithm,m.model_mode FROM predictions p LEFT JOIN models m ON m.id=p.model_id "
                "WHERE p.prediction_mode=? ORDER BY p.created_at DESC LIMIT ?", (mode, limit)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT p.*,m.scenario,m.algorithm,m.model_mode FROM predictions p LEFT JOIN models m ON m.id=p.model_id "
                "ORDER BY p.created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def latest_prediction(mode=None, include_replay=False):
    with connect() as con:
        where = []
        params = []
        if mode:
            where.append('p.prediction_mode=?')
            params.append(mode)
        if not include_replay:
            where.append("COALESCE(p.source_type,'operational')!='historical_replay'")
        clause = (' WHERE ' + ' AND '.join(where)) if where else ''
        sql = (
            "SELECT p.*,m.scenario,m.algorithm,m.model_mode FROM predictions p "
            "LEFT JOIN models m ON m.id=p.model_id" + clause +
            " ORDER BY p.created_at DESC LIMIT 1"
        )
        r = con.execute(sql, tuple(params)).fetchone()
        return dict(r) if r else None


def delete_prediction(prediction_id):
    with connect() as con:
        con.execute("DELETE FROM predictions WHERE id=?", (prediction_id,))
    log('delete', 'prediction', prediction_id)


# ------------------------------ DASHBOARD -----------------------------------
def dashboard_stats():
    with connect() as con:
        return {
            'datasets': con.execute('SELECT COUNT(*) FROM datasets').fetchone()[0],
            'experiments': con.execute('SELECT COUNT(*) FROM experiments').fetchone()[0],
            'models': con.execute('SELECT COUNT(*) FROM models').fetchone()[0],
            'predictions': con.execute('SELECT COUNT(*) FROM predictions').fetchone()[0],
        }


def list_activity_logs(limit=100):
    with connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM activity_logs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()]


# --------------------------- PARITY VALIDATION -----------------------------
def save_parity_report(dataset_id, reference_id, mode, report):
    pid = _id('PAR')
    status = 'completed' if not report.get('aborted') else 'aborted'
    passed = int(bool(report.get('passed')))
    with connect() as con:
        con.execute(
            "INSERT INTO parity_reports(id,dataset_id,reference_id,mode,status,passed,report_json) VALUES(?,?,?,?,?,?,?)",
            (pid, dataset_id, reference_id, mode, status, passed, json.dumps(report, default=str)),
        )
    log('create', 'parity_report', pid, f'{reference_id} • {status}')
    return pid


def list_parity_reports(limit=50):
    with connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT p.*,d.name dataset_name FROM parity_reports p JOIN datasets d ON d.id=p.dataset_id "
            "ORDER BY p.created_at DESC LIMIT ?", (limit,)
        ).fetchall()]


def get_parity_report(report_id):
    with connect() as con:
        r = con.execute(
            "SELECT p.*,d.name dataset_name FROM parity_reports p JOIN datasets d ON d.id=p.dataset_id WHERE p.id=?",
            (report_id,),
        ).fetchone()
    if not r: return None
    out = dict(r)
    try: out['report'] = json.loads(out['report_json'])
    except Exception: out['report'] = {}
    return out


def delete_parity_report(report_id):
    with connect() as con:
        con.execute("DELETE FROM parity_reports WHERE id=?", (report_id,))
    log('delete', 'parity_report', report_id)


def latest_parity_report():
    with connect() as con:
        r = con.execute(
            "SELECT p.*,d.name dataset_name FROM parity_reports p JOIN datasets d ON d.id=p.dataset_id "
            "ORDER BY p.created_at DESC LIMIT 1"
        ).fetchone()
    return dict(r) if r else None
