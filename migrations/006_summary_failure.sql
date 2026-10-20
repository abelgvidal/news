CREATE TABLE summary_failure (
    cluster_id INTEGER PRIMARY KEY REFERENCES cluster(id) ON DELETE CASCADE,
    n_members INTEGER NOT NULL,
    failed_at TEXT NOT NULL,
    reason TEXT NOT NULL
);
