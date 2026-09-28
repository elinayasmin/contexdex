-- Runs once, the first time the Postgres container starts.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id         SERIAL PRIMARY KEY,
    content    TEXT NOT NULL,
    source     TEXT NOT NULL,          -- file name or URL
    section    TEXT,                   -- heading the chunk came from
    category   TEXT,                   -- e.g. 'hr', 'it', 'finance' (used for metadata filtering)
    embedding  vector(384)             -- all-MiniLM-L6-v2 produces 384-dim vectors
);

-- Index for fast cosine-similarity search
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_category_idx ON chunks (category);
