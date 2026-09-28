"""Database connection helper shared by ingest, retrieval and the API."""
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


def get_conn(autocommit=False):
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=autocommit)
    register_vector(conn)
    return conn


if __name__ == "__main__":
    # Quick sanity check: python db.py
    with get_conn() as conn:
        ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        print(f"Connected. pgvector version: {ext[0]}, chunks stored: {count[0]}")
