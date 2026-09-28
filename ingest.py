"""Step 2: ingest PDFs and web pages -> section-aware chunks -> embeddings -> pgvector.

Where sources come from:
  data/<category>/*.pdf   the folder name becomes the chunk's category (e.g. data/hr/leave.pdf)
  data/*.pdf              category "general"
  data/urls.txt           one "<category> <url>" per line, '#' for comments

Run:
  python ingest.py          # (re)ingest every source; re-running replaces that source's chunks
  python ingest.py --reset  # wipe the table first
"""
import argparse
import os
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from db import get_conn

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
CHUNK_WORDS = 350     # ~500 tokens
OVERLAP_WORDS = 50    # repeated between neighbouring chunks so sentences aren't cut off from their context
MIN_WORDS = 10        # drop fragments too short to be useful


# ---------- loading: each loader returns a list of (section, text) ----------

def looks_like_heading(line):
    """Numbered ("2.1 Leave Policy") or ALL-CAPS short lines are treated as headings."""
    line = line.strip()
    if not line or len(line.split()) > 12:
        return False
    return line.isupper() or bool(re.match(r"^\d+(\.\d+)*\.?\s+[A-Za-z]", line))


def read_pdf(path):
    sections, heading, buf, start_page = [], None, [], 1
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        for line in (page.extract_text() or "").splitlines():
            if looks_like_heading(line):
                if buf:
                    sections.append((heading or f"page {start_page}", " ".join(buf)))
                    buf = []
                heading = line.strip()
            elif line.strip():
                if not buf:
                    start_page = page_no
                buf.append(line.strip())
    if buf:
        sections.append((heading or f"page {start_page}", " ".join(buf)))
    return sections


def read_url(url):
    resp = requests.get(url, timeout=20, headers={"User-Agent": "ContexDex/0.1"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    heading = soup.title.get_text(strip=True) if soup.title else url
    sections, buf = [], []
    for el in root.find_all(["h1", "h2", "h3", "p", "li", "pre"]):
        if el.name not in ("h1", "h2", "h3") and el.find_parent(["li", "p"]):
            continue  # nested element, already included via its parent
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in ("h1", "h2", "h3"):
            if buf:
                sections.append((heading, " ".join(buf)))
                buf = []
            heading = text
        else:
            buf.append(text)
    if buf:
        sections.append((heading, " ".join(buf)))
    return sections


def collect_sources():
    """Yield (category, source_name, loader) for every PDF and URL."""
    for pdf in sorted(DATA_DIR.rglob("*.pdf")):
        category = "general" if pdf.parent == DATA_DIR else pdf.parent.name
        yield category, pdf.relative_to(DATA_DIR).as_posix(), lambda p=pdf: read_pdf(p)

    urls_file = DATA_DIR / "urls.txt"
    if urls_file.exists():
        for line in urls_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            category, url = line.split(maxsplit=1)
            yield category, url, lambda u=url: read_url(u)


# ---------- chunking ----------

def chunk_words(text):
    """Split text into overlapping windows of CHUNK_WORDS words."""
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, max(len(words) - OVERLAP_WORDS, 1), step):
        yield " ".join(words[start:start + CHUNK_WORDS])


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete all chunks before ingesting")
    args = parser.parse_args()

    model = SentenceTransformer(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    with get_conn() as conn:
        if args.reset:
            conn.execute("TRUNCATE chunks RESTART IDENTITY")

        total = 0
        for category, source, load in collect_sources():
            try:
                sections = load()
            except Exception as e:
                print(f"  skipped {source}: {e}")
                continue

            rows = [(section, chunk)
                    for section, text in sections
                    for chunk in chunk_words(text)
                    if len(chunk.split()) >= MIN_WORDS]
            if not rows:
                print(f"  skipped {source}: no text extracted")
                continue

            # Prefix the section heading so the embedding knows what the chunk is about.
            vectors = model.encode([f"{s}: {c}" for s, c in rows],
                                   batch_size=32, normalize_embeddings=True)

            conn.execute("DELETE FROM chunks WHERE source = %s", (source,))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (content, source, section, category, embedding) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    [(c, source, s, category, v) for (s, c), v in zip(rows, vectors)],
                )
            conn.commit()
            total += len(rows)
            print(f"  {source}: {len(rows)} chunks [{category}]")

        print(f"Done. {total} chunks ingested.")


if __name__ == "__main__":
    main()
