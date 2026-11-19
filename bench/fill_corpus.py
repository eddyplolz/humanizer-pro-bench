#!/usr/bin/env python3
"""Fill the gaps that humanizer-pro's own fetch leaves on a GitHub runner.

Two gaps showed up on the first full Prepare run:

- Gutenberg answers 503 to GitHub's runners, so `build-essays` fails. This
  tries mirrors in order and keeps only the chunks whose digest matches the
  pinned manifest, so the human pool is the same text the public repo lists.
- `corpus.py fetch` asks Stack Exchange for up to 100 answers per call, but
  the API returns 30 per page unless told otherwise, so half the answers
  came back missing. This asks again with pagesize=100.

Nothing here edits the manifest. It only writes cache files whose text
matches a manifest digest, and it always exits 0 after printing what it
filled, so a flaky site never blocks the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rewrite_bench as bench

corpus = bench.corpus

GUTENBERG_MIRRORS = (
    "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
    "https://gutenberg.pglaf.org/cache/epub/{gid}/pg{gid}.txt",
    "https://aleph.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
    "https://mirrors.xmission.com/gutenberg/cache/epub/{gid}/pg{gid}.txt",
    "https://gutenberg.pglaf.org/{path}/{gid}/{gid}-0.txt",
    "https://aleph.gutenberg.org/{path}/{gid}/{gid}-0.txt",
)


def gutenberg_dir(gid: int) -> str:
    """Mirror directory for a book id: 205 -> '2/0', 2944 -> '2/9/4'."""
    digits = str(gid)
    return "/".join(digits[:-1]) if len(digits) > 1 else "0"


def missing(kind: str) -> list[dict]:
    return [
        e for e in corpus.load_manifest()["entries"]
        if e["kind"] == kind and not corpus.cache_path(e["id"]).exists()
    ]


def fill_gutenberg(sleep: float, fetch=None) -> tuple[int, int]:
    """(filled, wanted) for the Gutenberg essay chunks."""
    fetch = fetch or corpus.fetch_text
    wanted = missing("gutenberg-work")
    by_digest = {e["sha256"]: e for e in wanted}
    filled = 0
    for gid in sorted({int(e["source"]["gutenberg_id"]) for e in wanted}):
        book_filled = 0
        for template in GUTENBERG_MIRRORS:
            url = template.format(gid=gid, path=gutenberg_dir(gid))
            try:
                raw = fetch(url, sleep)
            except (*corpus.FETCH_ERRORS, ValueError) as error:
                print(f"gutenberg {gid}: {url} failed ({error})")
                continue
            for chunk in corpus.gutenberg_chunks(raw):
                entry = by_digest.pop(corpus.sha256_text(chunk), None)
                if entry is not None:
                    corpus.write_cache(entry["id"], chunk)
                    book_filled += 1
            print(f"gutenberg {gid}: {url} gave {book_filled} matching chunks")
            if not any(int(e["source"]["gutenberg_id"]) == gid for e in by_digest.values()):
                break
        filled += book_filled
    return filled, len(wanted)


def fill_stackexchange(sleep: float, get=None) -> tuple[int, int]:
    """(filled, wanted) for the Stack Exchange answers."""
    get = get or corpus.stackexchange_get
    wanted = missing("stackexchange-answer")
    by_site: dict[str, list[dict]] = {}
    for entry in wanted:
        by_site.setdefault(entry["source"]["site"], []).append(entry)
    filled = 0
    for site, entries in sorted(by_site.items()):
        for start in range(0, len(entries), 100):
            batch = entries[start : start + 100]
            ids = ";".join(str(e["source"]["answer_id"]) for e in batch)
            try:
                payload = get(f"answers/{ids}", {"site": site, "pagesize": 100}, sleep)
            except (*corpus.FETCH_ERRORS, ValueError) as error:
                print(f"stackexchange {site}: failed ({error})")
                continue
            by_id = {int(e["source"]["answer_id"]): e for e in batch}
            for item in payload.get("items", []):
                entry = by_id.get(int(item.get("answer_id", 0)))
                if entry is None or entry.get("extraction") != corpus.HTML_STRIP_VERSION:
                    continue
                corpus.write_cache(entry["id"], corpus.html_strip(str(item.get("body") or "")))
                filled += 1
    return filled, len(wanted)


def main() -> int:
    got, want = fill_stackexchange(sleep=1.0)
    print(f"stack exchange: filled {got} of {want} missing answers")
    got, want = fill_gutenberg(sleep=3.0)
    print(f"gutenberg: filled {got} of {want} missing chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
