import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "dashboard.sqlite3"
CSV_DIR = Path(__file__).resolve().parent.parent / "data" / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            period TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            csv_path TEXT NOT NULL,
            sha1 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_datasets_project ON datasets(project);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_datasets_period ON datasets(period);")
    conn.commit()
    conn.close()

def add_dataset(
    dataset_id: str,
    project: str,
    period: str,
    original_filename: str,
    csv_path: str,
    sha1: str,
    row_count: int,
) -> None:
    init_db()
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO datasets (id, project, period, original_filename, csv_path, sha1, row_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (dataset_id, project, period, original_filename, csv_path, sha1, row_count, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()

def list_datasets(project: Optional[str] = None) -> List[Dict[str, Any]]:
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    if project:
        cur.execute(
            "SELECT * FROM datasets WHERE project=? ORDER BY period DESC, created_at DESC",
            (project,),
        )
    else:
        cur.execute("SELECT * FROM datasets ORDER BY created_at DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def delete_dataset(dataset_id: str) -> None:
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT csv_path FROM datasets WHERE id=?", (dataset_id,))
    row = cur.fetchone()
    if row:
        try:
            Path(row["csv_path"]).unlink(missing_ok=True)
        except Exception:
            pass
    cur.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
    conn.commit()
    conn.close()
