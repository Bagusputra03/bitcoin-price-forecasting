from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parent / "bagsistem.db"


def connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def _ensure_column(con, table, column, ddl):
    cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db():
    with connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, symbol TEXT NOT NULL,
            start_date TEXT NOT NULL, end_date TEXT NOT NULL, interval TEXT NOT NULL,
            description TEXT DEFAULT '', is_locked INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id TEXT NOT NULL, date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            UNIQUE(dataset_id,date), FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, experiment_type TEXT NOT NULL, dataset_id TEXT NOT NULL,
            status TEXT DEFAULT 'draft', notes TEXT DEFAULT '', config_json TEXT DEFAULT '{}',
            started_at TEXT, completed_at TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS experiment_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT NOT NULL, scenario TEXT, model_type TEXT,
            run_number INTEGER, seed INTEGER, mae REAL, rmse REAL, mape REAL, execution_time REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS fold_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_run_id INTEGER NOT NULL, fold_number INTEGER,
            train_size INTEGER, test_size INTEGER, n_hidden INTEGER, c_value REAL, mae REAL, rmse REAL, mape REAL,
            FOREIGN KEY(experiment_run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY, experiment_id TEXT, scenario TEXT, algorithm TEXT, n_hidden INTEGER, c_value REAL,
            feature_set TEXT, mape REAL, model_path TEXT, status TEXT DEFAULT 'inactive', created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY, model_id TEXT, prediction_date TEXT, last_close REAL, predicted_close REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(model_id) REFERENCES models(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL, object_type TEXT, object_id TEXT,
            description TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS parity_reports (
            id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, reference_id TEXT NOT NULL, mode TEXT NOT NULL,
            status TEXT NOT NULL, passed INTEGER DEFAULT 0, report_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
        );
        """)

        # Backward-compatible migrations. Existing local databases remain usable.
        _ensure_column(con, "datasets", "mode", "TEXT DEFAULT 'research'")
        _ensure_column(con, "datasets", "source", "TEXT DEFAULT 'Yahoo Finance'")
        _ensure_column(con, "datasets", "last_synced_at", "TEXT")

        _ensure_column(con, "experiment_runs", "n_hidden", "INTEGER")
        _ensure_column(con, "experiment_runs", "c_value", "REAL")

        _ensure_column(con, "models", "dataset_id", "TEXT")
        _ensure_column(con, "models", "trained_until", "TEXT")
        _ensure_column(con, "models", "config_json", "TEXT DEFAULT '{}'")
        _ensure_column(con, "models", "notes", "TEXT DEFAULT ''")
        _ensure_column(con, "models", "model_mode", "TEXT DEFAULT 'research'")
        _ensure_column(con, "models", "source_latest_date", "TEXT")

        _ensure_column(con, "predictions", "input_date", "TEXT")
        _ensure_column(con, "predictions", "delta_usd", "REAL")
        _ensure_column(con, "predictions", "delta_pct", "REAL")
        _ensure_column(con, "predictions", "feature_json", "TEXT DEFAULT '{}'")
        _ensure_column(con, "predictions", "prediction_mode", "TEXT DEFAULT 'research'")
        _ensure_column(con, "predictions", "data_freshness_date", "TEXT")
        _ensure_column(con, "predictions", "source_type", "TEXT DEFAULT 'operational'")

        # Normalize legacy rows created before mode separation.
        con.execute("UPDATE datasets SET mode='research' WHERE mode IS NULL OR mode=''")
        con.execute("UPDATE models SET model_mode='research' WHERE model_mode IS NULL OR model_mode=''")
        con.execute("UPDATE predictions SET prediction_mode='research' WHERE prediction_mode IS NULL OR prediction_mode=''")
        con.execute("UPDATE predictions SET source_type='operational' WHERE source_type IS NULL OR source_type=''")
