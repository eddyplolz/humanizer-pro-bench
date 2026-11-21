"""Summarize plan use of subscription rewrite runs, as Markdown.

Reads every rewrite file in a folder and reports, for rows made through a
subscription CLI: documents, requests, tokens, median time per document, and
the last plan-limit reading Codex reported. Claude Code reports what the same
requests would have cost on the API ("API-equivalent"); nobody pays that, but
it is the best single measure of how much of a plan limit a run used.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def summarize(folder: Path) -> str:
    lines = [
        "| File | Docs | Requests | Input tokens (cached) | Output tokens | API-equivalent | Median s/doc |",
        "|---|---|---|---|---|---|---|",
    ]
    limits = []
    for path in sorted(folder.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [row for row in rows if row.get("backend", "openrouter") != "openrouter"]
        if not rows:
            continue
        usage = [u for row in rows for u in row.get("usage", [])]
        prompt = sum(u.get("prompt_tokens") or 0 for u in usage)
        cached = sum(u.get("cached_tokens") or 0 for u in usage)
        out = sum(u.get("completion_tokens") or 0 for u in usage)
        equiv = sum(u.get("api_equivalent_usd") or 0 for u in usage)
        secs = statistics.median(row.get("seconds") or 0 for row in rows)
        limits += [u["rate_limits"] for u in usage if u.get("rate_limits")]
        lines.append(
            f"| {path.name} | {len(rows)} | {len(usage)} | {prompt:,} ({cached:,}) | {out:,} | "
            f"{f'${equiv:.2f}' if equiv else 'n/a'} | {secs:.0f} |"
        )
    if limits:
        lines += ["", f"Last Codex plan-limit reading: `{json.dumps(limits[-1])}`"]
    return "\n".join(["## Subscription plan use", "", *lines]) + "\n"


if __name__ == "__main__":
    print(summarize(Path(sys.argv[1] if len(sys.argv) > 1 else "data/rewrites")))
