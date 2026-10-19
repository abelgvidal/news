CREATE TABLE embedding (
    article_id INTEGER PRIMARY KEY REFERENCES article(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE cluster (
    id INTEGER PRIMARY KEY,
    model TEXT NOT NULL,
    vector_sum BLOB NOT NULL,
    size INTEGER NOT NULL,
    entities TEXT NOT NULL DEFAULT '[]',
    feed_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    last_published_at TEXT NOT NULL
);
CREATE INDEX cluster_last_published ON cluster(last_published_at);

CREATE TABLE cluster_member (
    article_id INTEGER PRIMARY KEY REFERENCES article(id) ON DELETE CASCADE,
    cluster_id INTEGER NOT NULL REFERENCES cluster(id) ON DELETE CASCADE,
    similarity REAL,
    runner_up_cluster_id INTEGER,
    runner_up_similarity REAL,
    note TEXT
);
CREATE INDEX cluster_member_cluster ON cluster_member(cluster_id);
