#!/usr/bin/env python3
"""Stop before any spend if OpenRouter doesn't list a requested model id.

Reads OpenRouter's public model list (no key needed) and exits 1 naming
each id it doesn't find, with the closest listed ids as hints.
"""

from __future__ import annotations

import difflib
import json
import sys
import urllib.request

MODELS_URL = "https://openrouter.ai/api/v1/models"


def missing(requested: list[str], listed: set[str]) -> list[str]:
    return [model for model in requested if model not in listed]


def main(argv: list[str]) -> int:
    with urllib.request.urlopen(MODELS_URL, timeout=60) as response:
        listed = {item["id"] for item in json.load(response)["data"]}
    bad = missing(argv, listed)
    for model in argv:
        if model not in bad:
            print(f"ok: {model}")
    for model in bad:
        hints = ", ".join(difflib.get_close_matches(model, sorted(listed), n=4, cutoff=0.5)) or "none"
        print(f"FAIL: OpenRouter doesn't list {model}. Closest: {hints}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
