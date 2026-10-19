CREATE TABLE feed (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    etag TEXT,
    last_modified TEXT,
    last_success_at TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);

CREATE TABLE article (
    id INTEGER PRIMARY KEY,
    feed_id INTEGER NOT NULL REFERENCES feed(id),
    guid TEXT,
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    image_url TEXT,
    published_at TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE INDEX article_published ON article(published_at);
CREATE UNIQUE INDEX article_feed_guid ON article(feed_id, guid) WHERE guid IS NOT NULL;
