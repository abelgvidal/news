CREATE TABLE summary (
    cluster_id INTEGER PRIMARY KEY REFERENCES cluster(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    n_members INTEGER NOT NULL,
    headline TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
