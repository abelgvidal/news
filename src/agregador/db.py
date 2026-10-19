import sqlite3
from pathlib import Path

RETENTION_DAYS = 30
MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def migrate(conn: sqlite3.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> int:
    """Apply numbered *.sql files not yet applied. Returns the schema version."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for sql_file in sorted(migrations_dir.glob("*.sql")):
        version = int(sql_file.name.split("_", 1)[0])
        if version <= current:
            continue
        conn.executescript(sql_file.read_text())
        conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()
        current = version
    return current


def purge_old(conn: sqlite3.Connection, days: int = RETENTION_DAYS) -> int:
    """Delete articles older than `days`, plus clusters left without members."""
    from datetime import UTC, datetime, timedelta

    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    n = conn.execute("DELETE FROM article WHERE published_at < ?", (cutoff,)).rowcount
    conn.execute("DELETE FROM cluster WHERE id NOT IN (SELECT cluster_id FROM cluster_member)")
    conn.commit()
    return n
