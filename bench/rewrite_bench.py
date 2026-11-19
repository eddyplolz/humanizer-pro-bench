#!/usr/bin/env python3
"""Run the rewriter over a fixed sample of machine text, for the rewriter benchmark.

Runs against a checkout of humanizer-pro (HP_ROOT, default ./upstream). The
rewriter is SKILL.md: instructions a model follows. This script has a model
follow them on a fixed sample of machine test documents, so that later changes
to SKILL.md can be measured against the same input. detect_bench.py scores the
results with open-source detectors.

Arms (each rewrites the same sample):
- `skill`: SKILL.md plus the references Quick rewrite mode uses, and the
  request "Humanize this". The model cannot run tools here, so the script runs
  the audit and compare mode on each draft and sends the findings back, as
  SKILL.md tells an agent to do, for up to --passes drafts in all.
- `control`: the same model with one plain request and no skill. The skill
  has to beat this arm, or it adds nothing to the model alone.
- `rival`: another skill's SKILL.md (--rival-skill PATH), one pass, since a
  rival ships no audit to loop on.

Requests go through OpenRouter (OPENROUTER_API_KEY), as in generate_machine.py:
no fallback models, and every result records the model that served it. A run
prints an estimated cost and a ceiling before it sends anything; --dry-run
stops there. Each settled document is appended to a checkpoint in
data/rewrites/, so the same command resumes after a stop. --max-spend stops
the run before a request that could take the spend past the cap, counting
each reply's real token use at OpenRouter's listed prices.

Generations are not reproducible byte for byte. The sample, the prompts, and
this script are what make the measurement repeatable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("HP_ROOT", REPO / "upstream")).resolve()
SCRIPTS = ROOT / "scripts"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


corpus = _load("corpus")
gm = _load("generate_machine")
audit = _load("humanizer_audit")

DATA_DIR = REPO / "data"
BENCH_DIR = DATA_DIR / "rewrites"

# Per split: (group, kind, register, count). "fresh" is the headline group:
# every document of the generated pool (current models answering the public
# prompts; count None = all of them), split by prompt so both models' answers
# to one prompt land on the same side. RAID is reported apart, because at
# least one detector (Desklib) was trained on RAID's data. WildChat is left
# out: it can need a Hugging Face login, and this repo keeps one secret.
SAMPLE_PLAN = {
    "test": (
        [("fresh", "generated", None, None)]
        + [("raid", "raid-generation", register, 10) for register in ("chat", "news", "wiki")]
    ),
    "dev": (
        [("fresh", "generated", None, None)]
        + [("raid", "raid-generation", register, 6) for register in ("chat", "news", "wiki")]
    ),
}
OPTIONAL_KINDS: tuple[str, ...] = ()

# The references SKILL.md's Quick rewrite mode leans on: the artifact sweep it
# runs first, the full tell catalog, and the register table.
SKILL_REFERENCES = ("reference/llm-artifacts.md", "reference/tell-catalog.md", "reference/registers.md")
HARNESS_NOTE = (
    "You are running inside a test harness with no tools. You cannot run the audit CLI or "
    "compare mode yourself: after each draft, the user pastes their results for you. "
    "Reply with only the full rewritten text: no title, no notes, no audit, no code fence."
)
SKILL_REQUEST = "Humanize this:\n\n"
FEEDBACK_HEAD = (
    "Here are the audit and compare-mode results for your draft. Revise to fix them. "
    "Reply with only the full revised text.\n\n"
)
CONTROL_PROMPT = (
    "Rewrite this so it reads like a person wrote it. Keep every fact. "
    "Reply with only the rewritten text.\n\n"
)
RIVAL_REQUEST = SKILL_REQUEST

DEFAULT_MODELS = ("anthropic/claude-sonnet-5.5",)
DEFAULT_PASSES = 3
MAX_TOKENS = 16000
CHARS_PER_TOKEN = gm.CHARS_PER_TOKEN
# Claude's tokenizer packs fewer characters per token than OpenAI's. The
# first paid skill run measured about 2.5 characters per token for Sonnet
# (28,900 prompt tokens for a 71,400-character request) against about 3.9
# for GPT-5.4, so a flat 4 put Sonnet's real cost above its "ceiling".
ANTHROPIC_CHARS_PER_TOKEN = 2.4


def chars_per_token(model: str) -> float:
    return ANTHROPIC_CHARS_PER_TOKEN if model.startswith("anthropic/") else CHARS_PER_TOKEN

# A first line that introduces the text instead of being part of it.
LEAD_IN_RE = re.compile(r"^(?:here(?:'s| is| are)|sure|certainly|of course|below is)\b[^\n]{0,120}:\s*$", re.IGNORECASE)
FENCE_RE = re.compile(r"^```[\w-]*\n(.*)\n```$", re.DOTALL)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -------------------------------------------------------------------- sample


def entry_split(entry: dict) -> str:
    """Generated entries split by prompt, everything else by digest."""
    prompt = (entry.get("source") or {}).get("prompt_id")
    if entry["kind"] == "generated" and prompt:
        return corpus.digest_split(corpus.sha256_text(prompt))
    return entry.get("split") or corpus.digest_split(entry["sha256"])


def bench_sample(entries: list[dict], split: str = "test", plan: dict = SAMPLE_PLAN) -> list[dict]:
    """The fixed documents for a split, ordered by a hash of the entry id (as
    generate_machine.polish_sample does), so the sample is set before any
    rewrite is scored. Each item carries id, group, kind, register, split."""
    sample = []
    for group, kind, register, count in plan[split]:
        pool = [
            entry for entry in entries
            if entry["kind"] == kind and register in (None, entry["register"])
            and entry.get("label") == "machine" and entry_split(entry) == split
        ]
        pool.sort(key=lambda entry: corpus.sha256_text(entry["id"]))
        if not pool and kind in OPTIONAL_KINDS:
            continue
        if count is not None and len(pool) < count:
            raise ValueError(f"{split}: only {len(pool)} {kind}/{register} documents, need {count}")
        sample.extend(
            {"id": e["id"], "group": group, "kind": kind, "register": e["register"], "split": split, "model": e.get("model")}
            for e in pool[:count]
        )
    return sample


def read_text(entry_id: str) -> str:
    path = corpus.cache_path(entry_id)
    if not path.exists():
        raise FileNotFoundError(entry_id)
    return path.read_bytes().decode("utf-8")


# ------------------------------------------------------------------- prompts


def skill_system(root: Path = ROOT) -> str:
    parts = [(root / "SKILL.md").read_text(encoding="utf-8")]
    for rel in SKILL_REFERENCES:
        parts.append(f"<file path=\"{rel}\">\n{(root / rel).read_text(encoding='utf-8')}\n</file>")
    parts.append(HARNESS_NOTE)
    return "\n\n".join(parts)


def rival_system(path: Path) -> str:
    return f"{path.read_text(encoding='utf-8')}\n\n{HARNESS_NOTE}"


def clean_reply(text: str) -> tuple[str, bool]:
    """Strip a code fence around the whole reply and a lead-in line such as
    "Here is the rewrite:". Returns (text, stripped?) so the count is reported."""
    stripped = False
    body = text.strip()
    fence = FENCE_RE.match(body)
    if fence:
        body, stripped = fence.group(1).strip(), True
    lines = body.split("\n", 1)
    if len(lines) == 2 and LEAD_IN_RE.match(lines[0].strip()):
        body, stripped = lines[1].strip(), True
    return body, stripped


def blocking(finding: dict) -> bool:
    return finding["severity"] in ("warning", "error")


def feedback_for(original: str, draft: str) -> tuple[str, dict]:
    """Audit and compare results for a draft, as the user message an agent
    would see after running the CLI. Returns (message, state); state["clean"]
    is True when the audit has no warning or error and compare has no error.
    A compare.meaning warning alone does not force another draft: SKILL.md
    says to reread that pair and keep the edit if the meaning held."""
    result = audit.audit_text(draft, "draft.md")
    compare = audit.compare_texts(original, draft, "original.md", "draft.md")
    audit_hits = [f for f in result["findings"] if blocking(f)]
    compare_hits = list(compare["findings"])
    state = {
        "risk": int(result["risk_score"]),
        "audit_hits": len(audit_hits),
        "compare_errors": sum(1 for f in compare_hits if f["severity"] == "error"),
        "compare_warnings": sum(1 for f in compare_hits if f["severity"] != "error"),
        "clean": not audit_hits and not any(f["severity"] == "error" for f in compare_hits),
    }
    lines = [f"humanizer-audit draft.md: risk {state['risk']}"]
    lines += [f"  {f['severity'].upper()} L{f['line']} {f['id']}: {f['message']} [{f.get('evidence', '')}]" for f in audit_hits]
    lines.append("humanizer-audit --compare original.md draft.md:")
    lines += [f"  {f['severity'].upper()} {f['id']}: {f['message']} [{f.get('evidence', '')}]" for f in compare_hits]
    if not compare_hits:
        lines.append("  no findings")
    return FEEDBACK_HEAD + "\n".join(lines), state


# ---------------------------------------------------------------------- runs


class BenchClient(gm.OpenRouterClient):
    """OpenRouterClient that takes a full message list and keeps token usage."""

    def chat(self, model: str, max_tokens: int, messages: list[dict]):
        # usage.include asks OpenRouter to report each reply's cost.
        body = {"model": model, "max_tokens": max_tokens, "messages": messages, "usage": {"include": True}}
        attempt = 0
        while True:
            try:
                payload = self._post(body)
                return gm.parse_completion(payload), payload.get("usage") or {}
            except (gm.OpenRouterErrors.APIStatusError, gm.OpenRouterErrors.APIConnectionError) as error:
                if not gm._retryable(error) or attempt >= self._retries:
                    raise
            self._sleep(2**attempt)
            attempt += 1


def rewrite_one(client, model: str, arm: str, system: str | None, original: str, passes: int, max_tokens: int) -> dict:
    """Rewrite one document. Returns a record with every draft, the final
    text, the outcome, and token usage. Only the skill arm loops."""
    if arm == "control":
        messages = [{"role": "user", "content": CONTROL_PROMPT + original}]
    else:
        request = SKILL_REQUEST if arm == "skill" else RIVAL_REQUEST
        messages = [{"role": "system", "content": system}, {"role": "user", "content": request + original}]
    limit = passes if arm == "skill" else 1
    drafts, states, usage, served, stripped = [], [], [], None, 0
    outcome = "kept"
    for _ in range(limit):
        response, used = client.chat(model, max_tokens, messages)
        served = response.model
        usage.append({k: used.get(k) for k in ("prompt_tokens", "completion_tokens", "cost") if k in used})
        if response.stop_reason in ("refusal", "max_tokens"):
            outcome = "refused" if response.stop_reason == "refusal" else "truncated"
            break
        raw = gm.reply_text(response)
        draft, was_stripped = clean_reply(raw)
        stripped += was_stripped
        if not draft:
            outcome = "empty"
            break
        drafts.append(draft)
        message, state = feedback_for(original, draft)
        states.append(state)
        if arm != "skill" or state["clean"] or (len(drafts) > 1 and drafts[-1] == drafts[-2]):
            break
        messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": message}]
    return {
        "served_model": served,
        "outcome": outcome,
        "drafts": drafts,
        "final": drafts[-1] if drafts else None,
        "states": states,
        "usage": usage,
        "lead_in_stripped": stripped,
    }


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-")


def output_path(arm: str, model: str, split: str, tag: str = "") -> Path:
    suffix = f"-{tag}" if tag else ""
    return BENCH_DIR / f"rewrites-{split}-{arm}{suffix}-{model_slug(model)}.jsonl"


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["id"])
    return done


def draft_cost(sys_tokens: float, doc: float, reply: float, drafts: int, prices: tuple[float, float]) -> float:
    """USD for one document: draft k sees the instructions, the original,
    every earlier draft, and a feedback message (about 300 tokens) after each."""
    prompt_price, completion_price = prices
    cost = 0.0
    for k in range(drafts):
        cost += (sys_tokens + doc + k * (reply + 300) + gm.PROMPT_OVERHEAD_TOKENS) * prompt_price
        cost += reply * completion_price
    return cost


def estimate(
    items: list[dict], system: str | None, arm: str, passes: int, prices: tuple[float, float], cpt: float = CHARS_PER_TOKEN
) -> tuple[float, float]:
    """(estimate, ceiling) in USD. The estimate assumes two drafts for the
    skill arm and replies as long as the document; the ceiling assumes every
    allowed draft and replies 1.5 times the document."""
    sys_tokens = len(system or "") / cpt
    typical = min(2, passes) if arm == "skill" else 1
    most = passes if arm == "skill" else 1
    est = ceil = 0.0
    for item in items:
        doc = len(item["text"]) / cpt
        est += draft_cost(sys_tokens, doc, doc, typical, prices)
        ceil += draft_cost(sys_tokens, doc, 1.5 * doc, most, prices)
    return est, ceil


def spent_usd(usage: list[dict], prices: tuple[float, float]) -> float:
    """What a document's requests cost: OpenRouter's reported cost when the
    reply carries one, else its token counts at the listed prices."""
    total = 0.0
    for used in usage:
        if isinstance(used.get("cost"), (int, float)):
            total += used["cost"]
        else:
            total += (used.get("prompt_tokens") or 0) * prices[0] + (used.get("completion_tokens") or 0) * prices[1]
    return total


def worst_case_usd(
    item: dict, system: str | None, arm: str, passes: int, max_tokens: int, prices: tuple[float, float], cpt: float = CHARS_PER_TOKEN
) -> float:
    """The most one document can cost: every draft allowed, every reply at
    max_tokens. Checked before each document, so a run never passes its cap."""
    drafts = passes if arm == "skill" else 1
    sys_tokens = len(system or "") / cpt
    doc = len(item["text"]) / cpt
    return draft_cost(sys_tokens, doc, max_tokens, drafts, prices)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=["skill", "control", "rival"], required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="OpenRouter model ids.")
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES, help="Most drafts per document (skill arm).")
    parser.add_argument("--rival-skill", type=Path, help="Path to the rival's SKILL.md (rival arm).")
    parser.add_argument("--tag", default="", help="Label for a repeat run (noise check), kept in the file name.")
    parser.add_argument("--limit", type=int, help="Rewrite only the first N sampled documents.")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and cost; send nothing.")
    parser.add_argument("--max-spend", type=float, required=True, help="USD cap for this run, all models together.")
    args = parser.parse_args(argv)
    if args.arm == "rival" and not args.rival_skill:
        parser.error("--arm rival needs --rival-skill PATH")
    if args.tag and not re.fullmatch(r"[a-z0-9]+", args.tag):
        parser.error("--tag takes lowercase letters and digits only")

    sample = bench_sample(corpus.load_manifest()["entries"], args.split)[: args.limit]
    try:
        items = [{**item, "text": read_text(item["id"])} for item in sample]
    except FileNotFoundError as error:
        print(f"FAIL: {error.args[0]} is not in the cache; run corpus.py fetch first", file=sys.stderr)
        return 3
    system = skill_system() if args.arm == "skill" else rival_system(args.rival_skill) if args.arm == "rival" else None
    print(f"{len(items)} documents x {len(args.models)} model(s), arm {args.arm}, split {args.split}")
    if system:
        print(f"instructions: {len(system)} characters, sha256 {sha256_text(system)[:12]}")
    try:
        prices = gm.fetch_openrouter_prices()
    except Exception as error:  # noqa: BLE001 - any failure here means no estimate, so stop
        print(f"FAIL: could not read OpenRouter prices: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    total_est = total_ceil = 0.0
    for model in args.models:
        if model not in prices or min(prices[model]) < 0:
            print(f"FAIL: OpenRouter lists no fixed price for model {model!r}", file=sys.stderr)
            return 2
        todo = [item for item in items if item["id"] not in load_done(output_path(args.arm, model, args.split, args.tag))]
        est, ceil = estimate(todo, system, args.arm, args.passes, prices[model], chars_per_token(model))
        total_est += est
        total_ceil += ceil
        print(f"  {model}: {len(todo)} to do; estimated ${est:.2f}, ceiling ${ceil:.2f}")
    print(f"total: estimated ${total_est:.2f}, ceiling ${total_ceil:.2f} (hidden reasoning tokens not counted)")
    print(f"spend cap: ${args.max_spend:.2f}; the run stops before any document that could pass it")
    if args.dry_run:
        return 0
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("FAIL: set the OPENROUTER_API_KEY environment variable", file=sys.stderr)
        return 3
    client = BenchClient(api_key)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    errors = gm.OpenRouterErrors
    spent = 0.0
    for model in args.models:
        path = output_path(args.arm, model, args.split, args.tag)
        done = load_done(path)
        for item in items:
            if item["id"] in done:
                continue
            worst = worst_case_usd(item, system, args.arm, args.passes, args.max_tokens, prices[model], chars_per_token(model))
            if spent + worst > args.max_spend:
                print(
                    f"STOP: spent ${spent:.2f}; the next document could cost up to ${worst:.2f}, "
                    f"past the ${args.max_spend:.2f} cap. Finished documents are kept; run again to continue."
                )
                return 0
            try:
                record = rewrite_one(client, model, args.arm, system, item["text"], args.passes, args.max_tokens)
            except (errors.AuthenticationError, errors.PermissionDeniedError, errors.NotFoundError, errors.BadRequestError) as error:
                print(f"FAIL: {type(error).__name__}: {error.message}; finished documents are kept in {path}", file=sys.stderr)
                return 1
            except (errors.APIStatusError, errors.APIConnectionError) as error:
                print(f"{item['id']} on {model}: {type(error).__name__}; skipped (re-run to retry)", flush=True)
                continue
            spent += spent_usd(record["usage"], prices[model])
            row = {
                "id": item["id"], "group": item["group"], "register": item["register"], "split": item["split"],
                "source_model": item["model"], "arm": args.arm, "requested_model": model,
                "instructions_sha256": sha256_text(system) if system else None,
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record,
            }
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{item['id']} on {model}: {record['outcome']}, {len(record['drafts'])} draft(s)", flush=True)
        print(f"{model}: {len(load_done(path))} of {len(items)} documents in {path}")
    print(f"spent this run: ${spent:.2f} of ${args.max_spend:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
