#!/usr/bin/env python3
"""Score human text, machine text, and rewrites with open-source AI detectors.

A maintainer tool for the rewriter benchmark, not part of the audit, and not
It needs torch and transformers (requirements-detect.txt) and downloads
model weights from Hugging Face on first use.

Detectors (every score is oriented so higher means "more AI-like"):
- `desklib`: desklib/ai-text-detector-v1.01, a trained classifier
  (DeBERTa-v3-large). Its training data includes RAID, so RAID rows are
  reported apart.
- `fastdetectgpt`: zero-shot, one language model scoring how predictable
  each word is (Qwen2.5-0.5B by default, about 2 GB in memory).
- `binoculars`: zero-shot, two related language models loaded together, an
  observer (base) and a performer (instruct). `--pair tiny` (the default) is
  the Qwen2.5-0.5B pair, about 4 GB; `small` is the 1.5B pair; `falcon` is
  the published Falcon-7B pair, which needs a large GPU.

Built for a computer with little memory and no GPU: each run loads one
detector (`--detector`), scores one document at a time by default, and exits.
Run the detectors one after another, not together.

Each detector's cut-off is set on the human test split so that it flags 1% of
those documents (`--fpr`). "Detected" then means the same thing for every
detector. Scores are kept by text digest in data/scores/, so a re-run only
scores new text.

Subcommands:
- `score`: score the human test split, the sampled machine originals, and
  every rewrites-*.jsonl file in data/rewrites/.
- `report`: print detection rates, AUROC, and guardrails per arm and model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BENCH = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rewrite_bench", BENCH / "rewrite_bench.py")
bench = sys.modules.get("rewrite_bench") or importlib.util.module_from_spec(_spec)
if "rewrite_bench" not in sys.modules:
    sys.modules["rewrite_bench"] = bench
    _spec.loader.exec_module(bench)
_load = bench._load
corpus = _load("corpus")
fp = _load("fp_measure")
audit = _load("humanizer_audit")

DESKLIB_ID = "desklib/ai-text-detector-v1.01"
DESKLIB_MAX_TOKENS = 768
PAIRS = {
    "tiny": ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    "small": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    "falcon": ("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"),
}
FAST_MODEL = "Qwen/Qwen2.5-0.5B"
LM_MAX_TOKENS = 512
POSITION_CHUNK = 64
DEFAULT_FPR = 0.01
# Pools whose text is not public. Never scored here, even if a cache holds them.
PRIVATE_KINDS = ("persuade-essay",)
# Memory a run needs free before it loads anything, in GB: weights plus the
# working tensors for one document, with headroom. Binoculars loads two models.
NEEDS_GB = {"desklib": 3.0, "fastdetectgpt": 3.0, "binoculars": 5.0}


def available_gb() -> float | None:
    """Memory the system can hand out now (free plus reclaimable cache), in
    GB, or None when it cannot be read."""
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / 2**30
        return None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def memory_check(detector: str, available: float | None, override: float | None = None) -> str | None:
    """None when the run may start, else the reason it may not."""
    need = override if override is not None else NEEDS_GB[detector]
    if available is None:
        return f"cannot read free memory; {detector} needs {need:.0f} GB. Pass --min-free-gb 0 to run anyway"
    if available < need:
        return (
            f"{available:.1f} GB of memory is available and {detector} needs {need:.0f} GB. "
            "Close other programs (browsers first) and run again"
        )
    return None


# ------------------------------------------------------------- score math
#
# Kept free of torch at import time so the tests can check the math with
# plain tensors when torch is present and skip otherwise.


def pair_scores(observer_logits, performer_logits, input_ids, mask):
    """Binoculars and Fast-DetectGPT scores for one batch, higher = more AI.

    Binoculars (Hans et al., 2024): the performer's log-perplexity on the
    text divided by the cross-perplexity, the cross-entropy of the performer
    against the observer's next-token distribution. Low ratios mean machine
    text, so the score returned is the negative ratio.

    Fast-DetectGPT, analytic form (Bao et al., 2024): how far the text's
    log-likelihood under the scoring model (performer) sits above what the
    reference model's (observer) distribution expects, in standard
    deviations. High means machine text."""
    import torch

    labels = input_ids[:, 1:]
    m = mask[:, 1:].float()
    tokens = m.sum(-1).clamp(min=1)
    # Per-position sums, a chunk of positions at a time: full-vocabulary
    # float copies of every position at once would cost gigabytes.
    ll, cross, mean_ref, var_ref = [], [], [], []
    for start in range(0, labels.shape[1], POSITION_CHUNK):
        stop = start + POSITION_CHUNK
        perf_lp = torch.log_softmax(performer_logits[:, start:min(stop, labels.shape[1])].float(), -1)
        obs_p = torch.softmax(observer_logits[:, start:min(stop, labels.shape[1])].float(), -1)
        ll.append(perf_lp.gather(-1, labels[:, start:stop].unsqueeze(-1)).squeeze(-1))
        expected = (obs_p * perf_lp).sum(-1)
        cross.append(-expected)
        mean_ref.append(expected)
        var_ref.append((obs_p * perf_lp.pow(2)).sum(-1) - expected.pow(2))
        del perf_lp, obs_p
    ll, cross, mean_ref, var_ref = (torch.cat(part, -1) for part in (ll, cross, mean_ref, var_ref))
    log_ppl = -(ll * m).sum(-1) / tokens
    x_ppl = (cross * m).sum(-1) / tokens
    binoculars = -(log_ppl / x_ppl)
    fast = ((ll - mean_ref) * m).sum(-1) / ((var_ref * m).sum(-1).clamp(min=1e-9)).sqrt()
    return binoculars.tolist(), fast.tolist()


def threshold_at(human_scores: list[float], fpr: float) -> float:
    """The smallest cut-off that flags at most `fpr` of human documents
    (flagged means score > cut-off)."""
    ordered = sorted(human_scores)
    allowed = math.floor(fpr * len(ordered))
    return ordered[len(ordered) - allowed - 1]


def rate(scores: list[float], cut: float) -> tuple[int, int]:
    return sum(1 for s in scores if s > cut), len(scores)


# --------------------------------------------------------------- detectors


class Desklib:
    name = "desklib"

    def __init__(self, device: str, model_id: str = DESKLIB_ID):
        import torch
        from torch import nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

        class DesklibModel(PreTrainedModel):
            # The architecture from the model card: mean-pooled encoder
            # states into one logit.
            config_class = AutoConfig

            def __init__(self, config):
                super().__init__(config)
                self.model = AutoModel.from_config(config)
                self.classifier = nn.Linear(config.hidden_size, 1)
                self.post_init()

            def forward(self, input_ids, attention_mask):
                hidden = self.model(input_ids, attention_mask=attention_mask)[0]
                weights = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)
                return self.classifier(pooled).squeeze(-1)

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model, info = DesklibModel.from_pretrained(model_id, output_loading_info=True)
        if info.get("missing_keys"):
            # Random weights would score noise, so refuse to run.
            raise RuntimeError(f"{model_id}: weights missing for {sorted(info['missing_keys'])[:5]}")
        self.model = model.to(device).eval()
        self.device = device

    def score(self, texts: list[str]) -> dict[str, list[float]]:
        enc = self.tokenizer(texts, truncation=True, max_length=DESKLIB_MAX_TOKENS, padding=True, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            logits = self.model(enc["input_ids"], enc["attention_mask"])
        return {"desklib": self.torch.sigmoid(logits).tolist()}


def _causal_lm(model_id: str, device: str, bf16: bool):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if bf16 else torch.float32
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()


class FastDetectGPT:
    """White-box Fast-DetectGPT: one model is both reference and scorer."""

    name = "fastdetectgpt"

    def __init__(self, device: str, bf16: bool, model_id: str = FAST_MODEL):
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = _causal_lm(model_id, device, bf16)
        self.device = device

    def score(self, texts: list[str]) -> dict[str, list[float]]:
        enc = self.tokenizer(texts, truncation=True, max_length=LM_MAX_TOKENS, padding=True, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**enc).logits
        _binoculars, fast = pair_scores(logits, logits, enc["input_ids"], enc["attention_mask"])
        return {"fastdetectgpt": fast}


class Binoculars:
    def __init__(self, pair: str, device: str, bf16: bool):
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        observer_id, performer_id = PAIRS[pair]
        self.name = f"binoculars-{pair}"
        self.tokenizer = AutoTokenizer.from_pretrained(observer_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.observer = _causal_lm(observer_id, device, bf16)
        self.performer = _causal_lm(performer_id, device, bf16)
        self.device = device

    def score(self, texts: list[str]) -> dict[str, list[float]]:
        enc = self.tokenizer(texts, truncation=True, max_length=LM_MAX_TOKENS, padding=True, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            obs = self.observer(**enc).logits
            perf = self.performer(**enc).logits
        binoculars, _fast = pair_scores(obs, perf, enc["input_ids"], enc["attention_mask"])
        return {"binoculars": binoculars}


# ------------------------------------------------------------------- cache


SCORES_DIR = bench.DATA_DIR / "scores"


def cache_file(detector: str) -> Path:
    return SCORES_DIR / f"scores-{detector}.jsonl"


def load_scores(detector: str) -> dict[str, dict]:
    path = cache_file(detector)
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["sha256"]] = row["scores"]
    return out


def human_test_texts() -> list[tuple[str, str]]:
    """(id, text) for every cached human test document."""
    out = []
    for entry in corpus.load_manifest()["entries"]:
        if entry.get("label", "human") != "human" or entry["kind"] in PRIVATE_KINDS:
            continue
        if (entry.get("split") or corpus.digest_split(entry["sha256"])) != "test":
            continue
        path = corpus.cache_path(entry["id"])
        if not path.exists():
            continue
        text = path.read_bytes().decode("utf-8")
        # A source edited since the manifest was built is not the listed text.
        if corpus.sha256_text(text) == entry["sha256"]:
            out.append((entry["id"], text))
    return out


def rewrite_rows() -> list[dict]:
    rows = []
    for path in sorted(bench.BENCH_DIR.glob("rewrites-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append({**json.loads(line), "file": path.name})
    return rows


def all_texts(split: str, humans_only: bool = False) -> list[str]:
    texts = [text for _id, text in human_test_texts()]
    if humans_only:
        return texts
    texts += [bench.read_text(item["id"]) for item in bench.bench_sample(corpus.load_manifest()["entries"], split)]
    for row in rewrite_rows():
        texts += [draft for draft in row["drafts"]]
    return texts


def score_command(args: argparse.Namespace) -> int:
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    problem = memory_check(args.detector, available_gb(), args.min_free_gb)
    if problem:
        # Stop before loading any model, so a short machine is never pushed
        # into swapping.
        print(f"STOP: {problem}", file=sys.stderr)
        return 4
    if args.detector == "desklib":
        detectors = [Desklib(device)]
    elif args.detector == "fastdetectgpt":
        detectors = [FastDetectGPT(device, args.bf16)]
    else:
        detectors = [Binoculars(args.pair, device, args.bf16)]
    texts = list(dict.fromkeys(all_texts(args.split, args.humans_only)))
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    for detector in detectors:
        cached = load_scores(detector.name)
        todo = [t for t in texts if bench.sha256_text(t) not in cached]
        print(f"{detector.name}: {len(texts) - len(todo)} cached, {len(todo)} to score", flush=True)
        with cache_file(detector.name).open("a", encoding="utf-8", newline="\n") as handle:
            for start in range(0, len(todo), args.batch):
                batch = todo[start:start + args.batch]
                scores = detector.score(batch)
                for i, text in enumerate(batch):
                    row = {"sha256": bench.sha256_text(text), "scores": {k: v[i] for k, v in scores.items()}}
                    handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(f"  {min(start + args.batch, len(todo))}/{len(todo)}", flush=True)
    return 0


# ------------------------------------------------------------------ report


def guardrails(original: str, final: str) -> dict:
    compare = audit.compare_texts(original, final, "original.md", "final.md")
    result = audit.audit_text(final, "final.md")
    stats_hidden = sum(1 for f in result["findings"] if str(f["id"]).startswith("artifact.bypass"))
    return {
        "fact_drift": any(f["severity"] == "error" for f in compare["findings"]),
        "meaning_warning": any(str(f["id"]).startswith("compare.meaning") for f in compare["findings"]),
        "hidden": stats_hidden > 0,
        "length_ratio": corpus.word_count(final) / max(1, corpus.word_count(original)),
        "risk": int(result["risk_score"]),
    }


def pct(hits: int, total: int) -> str:
    if not total:
        return "-"
    low, high = fp.wilson_interval(hits, total)
    return f"{hits / total:.0%} ({low:.0%}-{high:.0%})"


def report_command(args: argparse.Namespace) -> int:
    scores = {}
    for name in ("desklib", "fastdetectgpt", f"binoculars-{args.pair}"):
        scores.update({k: {**scores.get(k, {}), **v} for k, v in load_scores(name).items()})
    if not scores:
        print("FAIL: no scores cached; run `detect_bench.py score` first", file=sys.stderr)
        return 3
    detectors = sorted({d for s in scores.values() for d in s})
    humans = [scores.get(bench.sha256_text(t)) for _id, t in human_test_texts()]
    humans = [h for h in humans if h]
    cuts = {d: threshold_at([h[d] for h in humans if d in h], args.fpr) for d in detectors}
    print(f"human test documents scored: {len(humans)}; cut-offs flag {args.fpr:.0%} of them")
    sample = bench.bench_sample(corpus.load_manifest()["entries"], args.split)
    originals = {item["id"]: bench.read_text(item["id"]) for item in sample}
    groups = {item["id"]: item["group"] for item in sample}

    # Rows: (label, group) -> list of (scores, guardrail dict or None)
    table = defaultdict(list)
    for item in sample:
        s = scores.get(bench.sha256_text(originals[item["id"]]))
        if s:
            table[("original (no rewrite)", item["group"])].append((s, None))
    for row in rewrite_rows():
        if row["split"] != args.split or row["id"] not in originals:
            continue
        if row.get("final"):
            s = scores.get(bench.sha256_text(row["final"]))
            if s:
                table[(f"{row['arm']} {row['requested_model']}", groups[row["id"]])].append((s, guardrails(originals[row["id"]], row["final"])))
            if row["arm"] == "skill" and row["drafts"]:
                s1 = scores.get(bench.sha256_text(row["drafts"][0]))
                if s1:
                    table[(f"skill, first draft only {row['requested_model']}", groups[row["id"]])].append(
                        (s1, guardrails(originals[row["id"]], row["drafts"][0]))
                    )
    header = ["Arm and model", "Group", "n"] + [f"{d} detected" for d in detectors] + [f"{d} AUROC" for d in detectors]
    header += ["fact drift", "meaning warn", "hidden chars", "median length"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for (label, group), rows in sorted(table.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        cells = [label, group, str(len(rows))]
        for d in detectors:
            cells.append(pct(*rate([r[0][d] for r in rows if d in r[0]], cuts[d])))
        for d in detectors:
            value = fp.value_auroc([h[d] for h in humans if d in h], [r[0][d] for r in rows if d in r[0]])
            cells.append("-" if value is None else f"{value:.2f}")
        guards = [g for _s, g in rows if g]
        if guards:
            cells.append(pct(sum(g["fact_drift"] for g in guards), len(guards)))
            cells.append(pct(sum(g["meaning_warning"] for g in guards), len(guards)))
            cells.append(str(sum(g["hidden"] for g in guards)))
            cells.append(f"{statistics.median(g['length_ratio'] for g in guards):.2f}")
        else:
            cells += ["-", "-", "-", "1.00"]
        print("| " + " | ".join(cells) + " |")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="Score every text the benchmark needs (cached).")
    score.add_argument("--detector", choices=["desklib", "fastdetectgpt", "binoculars"], required=True)
    score.add_argument("--pair", choices=sorted(PAIRS), default="tiny", help="Model pair for binoculars.")
    score.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    score.add_argument("--batch", type=int, default=1, help="Documents per forward pass; 1 keeps memory lowest.")
    score.add_argument("--min-free-gb", type=float, help="Override the free-memory check (GB).")
    score.add_argument("--bf16", action="store_true", help="Half the memory for the language models; slower on some CPUs.")
    score.add_argument("--split", choices=["test", "dev"], default="test")
    score.add_argument("--humans-only", action="store_true", help="Score only the human test set (no AI pool needed).")
    report = sub.add_parser("report", help="Print the benchmark tables.")
    report.add_argument("--pair", choices=sorted(PAIRS), default="tiny")
    report.add_argument("--fpr", type=float, default=DEFAULT_FPR)
    report.add_argument("--split", choices=["test", "dev"], default="test")
    args = parser.parse_args(argv)
    return score_command(args) if args.command == "score" else report_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
