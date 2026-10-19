ALTER TABLE feed ADD COLUMN country TEXT NOT NULL DEFAULT 'es';
ALTER TABLE cluster ADD COLUMN country TEXT NOT NULL DEFAULT 'es';
CREATE INDEX cluster_country ON cluster(country, last_published_at);
