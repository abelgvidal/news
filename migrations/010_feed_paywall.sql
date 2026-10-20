-- 1 = the outlet's articles often need a subscription (hard or metered paywall): shown with a lock
ALTER TABLE feed ADD COLUMN paywall INTEGER NOT NULL DEFAULT 0;
