#!/usr/bin/env python3
"""Keep the freshly generated machine pool in this repo between workflow runs.

generate_machine.py writes its pool into the humanizer-pro checkout, which is
thrown away after each run. `save` copies the pool's manifest entries and
texts to data/generated/; `restore` puts them back into a fresh checkout,
replacing the checkout's own generated entries (whose texts are not public).
Every text is checked against its digest both ways.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rewrite_bench as bench

corpus = bench.corpus
POOL_DIR = bench.DATA_DIR / "generated"
KIND = "generated"


def save(pool_dir: Path = POOL_DIR) -> int:
    entries = [e for e in corpus.load_manifest()["entries"] if e["kind"] == KIND]
    texts = {}
    for entry in entries:
        text = bench.read_text(entry["id"])
        if corpus.sha256_text(text) != entry["sha256"]:
            raise ValueError(f"{entry['id']}: cached text does not match its digest")
        texts[entry["id"]] = text
    if not entries:
        print("FAIL: the checkout has no generated pool to save", file=sys.stderr)
        return 1
    pool_dir.mkdir(parents=True, exist_ok=True)
    for old in pool_dir.glob("*.txt"):
        old.unlink()
    for entry_id, text in texts.items():
        (pool_dir / f"{entry_id}.txt").write_text(text, encoding="utf-8", newline="\n")
    (pool_dir / "entries.json").write_text(json.dumps({"entries": entries}, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(entries)} generated documents to {pool_dir}")
    return 0


def restore(pool_dir: Path = POOL_DIR) -> int:
    path = pool_dir / "entries.json"
    if not path.exists():
        print(f"FAIL: no saved pool at {path}; run the prepare workflow with generate on", file=sys.stderr)
        return 1
    saved = json.loads(path.read_text(encoding="utf-8"))["entries"]
    manifest = corpus.load_manifest()
    for entry in manifest["entries"]:
        if entry["kind"] == KIND:
            corpus.cache_path(entry["id"]).unlink(missing_ok=True)
    manifest["entries"] = [e for e in manifest["entries"] if e["kind"] != KIND] + saved
    for entry in saved:
        text = (pool_dir / f"{entry['id']}.txt").read_bytes().decode("utf-8")
        if corpus.sha256_text(text) != entry["sha256"]:
            raise ValueError(f"{entry['id']}: saved text does not match its digest")
        corpus.write_cache(entry["id"], text)
    corpus.save_manifest(manifest)
    print(f"restored {len(saved)} generated documents into the checkout")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["save", "restore"])
    args = parser.parse_args(argv)
    return save() if args.command == "save" else restore()


if __name__ == "__main__":
    raise SystemExit(main())
