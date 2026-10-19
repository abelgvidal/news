CREATE TABLE summary_run (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    attempted INTEGER NOT NULL,
    stored INTEGER NOT NULL,
    unverified INTEGER NOT NULL,
    gen_calls INTEGER NOT NULL,
    verify_calls INTEGER NOT NULL,
    escalations INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    tokens TEXT NOT NULL
);
CREATE INDEX summary_run_ts ON summary_run(ts);
