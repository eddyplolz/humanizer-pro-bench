from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HP_ROOT", str(REPO / "upstream"))


def load_module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "bench" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = load_module("rewrite_bench")
detect = load_module("detect_bench")
pool = load_module("pool")
fill = load_module("fill_corpus")
corpus = bench.corpus

AI_TEXT = (
    "In today's rapidly evolving digital landscape, effective collaboration serves as a crucial "
    "cornerstone for organizations seeking to unlock their full potential. It is important to note "
    "that this approach is not just about tools, but about creating a vibrant culture of innovation."
)
CLEAN_TEXT = (
    "Good collaboration depends less on the tool than on whether people know which decisions they "
    "own, where work is tracked, and how fast blockers get cleared."
)


# ------------------------------------------------------------------ sample


def test_sample_is_fixed_and_matches_the_plan() -> None:
    entries = corpus.load_manifest()["entries"]
    test = bench.bench_sample(entries, "test")
    assert test == bench.bench_sample(list(reversed(entries)), "test")
    groups = Counter(item["group"] for item in test)
    assert set(groups) == {"fresh", "raid"} and groups["raid"] == 30
    assert all(item["split"] == "test" for item in test)


def test_fresh_pool_splits_by_prompt() -> None:
    entries = [e for e in corpus.load_manifest()["entries"] if e["kind"] == "generated"]
    by_prompt = {}
    for entry in entries:
        by_prompt.setdefault(entry["source"]["prompt_id"], set()).add(bench.entry_split(entry))
    assert by_prompt and all(len(splits) == 1 for splits in by_prompt.values())


def test_dev_and_test_samples_never_share_a_document() -> None:
    entries = corpus.load_manifest()["entries"]
    test = {item["id"] for item in bench.bench_sample(entries, "test")}
    dev = {item["id"] for item in bench.bench_sample(entries, "dev")}
    assert dev and not test & dev


def test_sample_refuses_a_short_pool() -> None:
    plan = {"test": [("claude", "generated", "essay", 10_000)]}
    with pytest.raises(ValueError, match="need 10000"):
        bench.bench_sample(corpus.load_manifest()["entries"], "test", plan)


# ----------------------------------------------------------------- prompts


def test_skill_system_carries_skill_and_references() -> None:
    system = bench.skill_system()
    assert "# Humanizer Pro: Remove AI Writing Tells" in system
    for rel in bench.SKILL_REFERENCES:
        assert f'<file path="{rel}">' in system
    assert system.endswith(bench.HARNESS_NOTE)


@pytest.mark.parametrize(
    "reply, expected, stripped",
    [
        ("Plain text.", "Plain text.", False),
        ("Here is the rewrite:\nPlain text.", "Plain text.", True),
        ("```markdown\nPlain text.\n```", "Plain text.", True),
        ("Here is why it matters: plain text.", "Here is why it matters: plain text.", False),
    ],
)
def test_clean_reply(reply: str, expected: str, stripped: bool) -> None:
    assert bench.clean_reply(reply) == (expected, stripped)


def test_feedback_lists_audit_and_compare_findings() -> None:
    message, state = bench.feedback_for(AI_TEXT, AI_TEXT)
    assert not state["clean"] and state["audit_hits"] > 0
    assert "family3.filler_framing" in message
    message, state = bench.feedback_for("Sales rose 12% in 2021.", "Sales rose sharply.")
    assert state["compare_errors"] > 0
    assert "compare.number" in message


# -------------------------------------------------------------------- runs


class FakeClient:
    def __init__(self, replies: list[str], stop: str = "end_turn"):
        self.replies = list(replies)
        self.stop = stop
        self.calls: list[list[dict]] = []

    def chat(self, model: str, max_tokens: int, messages: list[dict]):
        self.calls.append([dict(m) for m in messages])
        text = self.replies.pop(0)
        response = SimpleNamespace(stop_reason=self.stop, content=[SimpleNamespace(type="text", text=text)], model=model + "-served")
        return response, {"prompt_tokens": 10, "completion_tokens": 5}


def test_skill_arm_loops_until_clean() -> None:
    client = FakeClient([AI_TEXT, CLEAN_TEXT, "never used"])
    record = bench.rewrite_one(client, "m", "skill", "SYSTEM", AI_TEXT, 3, 100)
    assert record["final"] == CLEAN_TEXT
    assert len(record["drafts"]) == 2 and len(client.calls) == 2
    second = client.calls[1]
    assert second[0] == {"role": "system", "content": "SYSTEM"}
    assert second[2]["role"] == "assistant" and second[3]["content"].startswith(bench.FEEDBACK_HEAD)
    assert record["states"][-1]["clean"]
    assert record["served_model"] == "m-served"


def test_skill_arm_stops_at_the_pass_limit_and_on_a_repeat() -> None:
    record = bench.rewrite_one(FakeClient([AI_TEXT] * 5), "m", "skill", "S", AI_TEXT, 3, 100)
    assert len(record["drafts"]) == 2  # second draft repeated the first
    other = AI_TEXT.replace("vibrant", "lively")
    record = bench.rewrite_one(FakeClient([AI_TEXT, other, AI_TEXT, other]), "m", "skill", "S", AI_TEXT, 3, 100)
    assert len(record["drafts"]) == 3


def test_control_and_rival_arms_send_one_request() -> None:
    client = FakeClient([AI_TEXT])
    record = bench.rewrite_one(client, "m", "control", None, AI_TEXT, 3, 100)
    assert len(client.calls) == 1 and client.calls[0][0]["content"].startswith(bench.CONTROL_PROMPT)
    client = FakeClient([AI_TEXT])
    bench.rewrite_one(client, "m", "rival", "RIVAL", AI_TEXT, 3, 100)
    assert len(client.calls) == 1 and client.calls[0][0]["content"] == "RIVAL"
    assert record["outcome"] == "kept"


def test_truncated_reply_keeps_no_draft() -> None:
    record = bench.rewrite_one(FakeClient(["cut"], stop="max_tokens"), "m", "skill", "S", AI_TEXT, 3, 100)
    assert record["outcome"] == "truncated" and record["final"] is None


def test_estimate_ceiling_covers_estimate() -> None:
    items = [{"text": AI_TEXT * 10}]
    est, ceil = bench.estimate(items, "x" * 40_000, "skill", 3, (1e-6, 5e-6))
    assert 0 < est < ceil
    control_est, _ = bench.estimate(items, None, "control", 3, (1e-6, 5e-6))
    assert control_est < est


def test_output_path_names_arm_model_and_tag() -> None:
    path = bench.output_path("skill", "anthropic/claude-sonnet-5.5", "test", "r2")
    assert path.name == "rewrites-test-skill-r2-anthropic-claude-sonnet-5.5.jsonl"
    assert path.parent == REPO / "data" / "rewrites"


# --------------------------------------------------------------- detectors


def test_threshold_flags_at_most_the_allowed_share() -> None:
    human = [float(i) for i in range(200)]
    cut = detect.threshold_at(human, 0.01)
    assert detect.rate(human, cut) == (2, 200)
    assert detect.rate([cut, cut + 1], cut) == (1, 2)


def test_pair_scores_rank_predictable_text_as_more_ai() -> None:
    torch = pytest.importorskip("torch")
    vocab, length = 8, 6
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    mask = torch.ones_like(ids)
    sharp = torch.full((1, length, vocab), -5.0)
    for pos in range(length - 1):
        sharp[0, pos, ids[0, pos + 1]] = 5.0  # the model expects each next token
    flat = torch.zeros((1, length, vocab))
    flat[0, :, 0] = 5.0  # the model expects token 0, which never comes
    b_sharp, f_sharp = detect.pair_scores(sharp, sharp, ids, mask)
    b_flat, f_flat = detect.pair_scores(flat, flat, ids, mask)
    assert f_sharp[0] > f_flat[0]
    assert b_sharp[0] > b_flat[0]


def test_report_prints_rates_guardrails_and_first_draft_rows(monkeypatch, capsys) -> None:
    humans = [(f"h{i}", f"Human text number {i} about gardens.") for i in range(100)]
    sample = [{"id": "m1", "group": "fresh", "register": "essay", "split": "test", "model": "x"}]
    final = "Sales rose sharply."
    rows = [{
        "id": "m1", "split": "test", "arm": "skill", "requested_model": "model-a", "file": "f",
        "drafts": [AI_TEXT, final], "final": final,
    }]
    texts = [t for _i, t in humans] + [AI_TEXT, final, "Sales rose 12% in 2021."]
    scores = {bench.sha256_text(t): {"desklib": float(i) / 1000} for i, t in enumerate(texts)}
    scores[bench.sha256_text("Sales rose 12% in 2021.")] = {"desklib": 0.99}
    monkeypatch.setattr(detect, "human_test_texts", lambda: humans)
    monkeypatch.setattr(detect, "rewrite_rows", lambda: rows)
    monkeypatch.setattr(detect, "load_scores", lambda name: scores if name == "desklib" else {})
    monkeypatch.setattr(bench, "bench_sample", lambda entries, split: sample)
    monkeypatch.setattr(bench, "read_text", lambda entry_id: "Sales rose 12% in 2021.")
    args = SimpleNamespace(pair="tiny", fpr=0.01, split="test")
    assert detect.report_command(args) == 0
    out = capsys.readouterr().out
    assert "human test documents scored: 100" in out
    assert "| original (no rewrite) | fresh | 1 | 100% (" in out
    assert "| skill model-a | fresh | 1 |" in out
    assert "| skill, first draft only model-a | fresh | 1 |" in out
    skill_row = next(line for line in out.splitlines() if line.startswith("| skill model-a"))
    assert "100% (" in skill_row.split("|")[-5]  # fact drift: the number 12% was lost


def test_pair_scores_chunking_matches_one_pass(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(7)
    obs = torch.randn((2, 9, 11), generator=gen)
    perf = torch.randn((2, 9, 11), generator=gen)
    ids = torch.randint(0, 11, (2, 9), generator=gen)
    mask = torch.ones_like(ids)
    mask[1, 7:] = 0
    monkeypatch.setattr(detect, "POSITION_CHUNK", 1000)
    whole = detect.pair_scores(obs, perf, ids, mask)
    monkeypatch.setattr(detect, "POSITION_CHUNK", 3)
    chunked = detect.pair_scores(obs, perf, ids, mask)
    for a, b in zip(whole, chunked):
        assert a == pytest.approx(b, rel=1e-5)


def test_memory_check_stops_a_short_machine() -> None:
    assert detect.memory_check("desklib", 8.0) is None
    assert "needs 5 GB" in detect.memory_check("binoculars", 4.0)
    assert "Close other programs" in detect.memory_check("fastdetectgpt", 0.9)
    assert "cannot read" in detect.memory_check("desklib", None)
    assert detect.memory_check("binoculars", 1.0, override=0) is None
    assert detect.available_gb() is None or detect.available_gb() > 0


def test_score_refuses_before_loading_a_model(monkeypatch, capsys) -> None:
    monkeypatch.setattr(detect, "available_gb", lambda: 0.9)
    monkeypatch.setattr(detect, "Desklib", lambda device: pytest.fail("model loaded"))
    args = SimpleNamespace(device="cpu", detector="desklib", min_free_gb=None, bf16=False, pair="tiny", batch=1, split="test")
    assert detect.score_command(args) == 4
    assert "STOP: 0.9 GB" in capsys.readouterr().err


# -------------------------------------------------------------- spend cap


def test_worst_case_bounds_real_spend() -> None:
    item = {"text": AI_TEXT * 5}
    prices = (3e-6, 15e-6)
    worst = bench.worst_case_usd(item, "x" * 80_000, "skill", 3, 16_000, prices)
    real = bench.spent_usd([{"prompt_tokens": 20_000, "completion_tokens": 800}] * 3, prices)
    assert real < worst
    assert bench.spent_usd([{"cost": 0.42, "prompt_tokens": 10**9}], prices) == 0.42


def test_run_stops_before_passing_the_cap(monkeypatch, tmp_path, capsys) -> None:
    items = [{"id": f"m{i}", "group": "fresh", "register": "essay", "split": "test", "model": "x"} for i in range(5)]
    monkeypatch.setattr(bench, "bench_sample", lambda entries, split: items)
    monkeypatch.setattr(bench, "read_text", lambda entry_id: AI_TEXT)
    monkeypatch.setattr(bench, "BENCH_DIR", tmp_path)
    monkeypatch.setattr(bench.gm, "fetch_openrouter_prices", lambda: {"m": (1e-6, 1e-6)})
    monkeypatch.setattr(bench.corpus, "load_manifest", lambda: {"entries": []})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = FakeClient([CLEAN_TEXT] * 10)

    def chat(model, max_tokens, messages):
        response, _ = FakeClient.chat(client, model, max_tokens, messages)
        return response, {"cost": 0.004}

    client.chat = chat
    monkeypatch.setattr(bench, "BenchClient", lambda key: client)
    # control arm, 1 request per document; worst case per document is
    # (prompt + 16,000 tokens) at $1 per million, about $0.016.
    assert bench.main(["--arm", "control", "--models", "m", "--max-spend", "0.03"]) == 0
    out = capsys.readouterr().out
    # Each document really costs $0.004; the run stops once spent plus the
    # $0.016 worst case would pass $0.03, so it never spends past the cap.
    rows = [json.loads(line) for line in (tmp_path / "rewrites-test-control-m.jsonl").read_text().splitlines()]
    spent = 0.004 * len(rows)
    assert "STOP: spent $" in out
    assert spent <= 0.03 and len(rows) < 5


# ------------------------------------------------------------------- pool


def test_pool_save_and_restore_round_trip(monkeypatch, tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(corpus, "CORPUS_DIR", corpus_dir)
    monkeypatch.setattr(corpus, "MANIFEST_PATH", corpus_dir / "manifest.json")
    monkeypatch.setattr(corpus, "CACHE_DIR", corpus_dir / "cache")
    pairs = [({"register": "essay", "author": "machine", "model": "m", "date": "2026-11", "extraction": "generated.v1"},
              AI_TEXT, {"prompt_id": "essay-01", "requested_model": "m"})]
    saved = corpus.save_pool("generated", pairs)
    pool_dir = tmp_path / "pool"
    assert pool.save(pool_dir) == 0
    # A fresh checkout: its own generated entry has no cached text.
    stale = {**saved[0], "id": "gen-000000000000", "sha256": "0" * 64}
    corpus.save_manifest({"entries": [stale]})
    (corpus_dir / "cache").joinpath(f"{saved[0]['id']}.txt").unlink()
    assert pool.restore(pool_dir) == 0
    entries = corpus.load_manifest()["entries"]
    assert [e["id"] for e in entries] == [saved[0]["id"]]
    assert bench.read_text(saved[0]["id"]) == AI_TEXT


def test_pool_restore_refuses_a_changed_text(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(corpus, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(corpus, "CACHE_DIR", tmp_path / "cache")
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    entry = {"id": "gen-0123456789ab", "kind": "generated", "sha256": corpus.sha256_text("original")}
    (pool_dir / "entries.json").write_text(json.dumps({"entries": [entry]}))
    (pool_dir / "gen-0123456789ab.txt").write_text("edited")
    with pytest.raises(ValueError, match="does not match"):
        pool.restore(pool_dir)


# ------------------------------------------------------------------- fill


def _manifest(monkeypatch, tmp_path, entries) -> None:
    for module in {corpus, detect.corpus, fill.corpus}:
        monkeypatch.setattr(module, "MANIFEST_PATH", tmp_path / "manifest.json")
        monkeypatch.setattr(module, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "manifest.json").write_text(json.dumps({"entries": entries}), encoding="utf-8")


def test_gutenberg_fill_tries_mirrors_and_keeps_only_listed_chunks(monkeypatch, tmp_path) -> None:
    para = " ".join(["word"] * 40)
    body = "\n\n".join(f"{para} {i}" for i in range(200))
    raw = f"header\n*** START OF THE BOOK ***\n{body}\n*** END OF THE BOOK ***\nfooter"
    chunks = corpus.gutenberg_chunks(raw)
    assert len(chunks) >= 2
    listed = [{"id": f"guten-{i:012x}", "kind": "gutenberg-work", "sha256": corpus.sha256_text(c),
               "source": {"gutenberg_id": 205}} for i, c in enumerate(chunks[:2])]
    _manifest(monkeypatch, tmp_path, listed)
    calls = []

    def fetch(url, _sleep):
        calls.append(url)
        if len(calls) == 1:
            raise corpus.urllib.error.URLError("503")
        return raw

    assert fill.fill_gutenberg(0, fetch) == (2, 2)
    assert len(calls) == 2 and "pglaf" in calls[1]
    assert bench.read_text(listed[0]["id"]) == chunks[0]
    assert len(list((tmp_path / "cache").glob("*.txt"))) == 2


def test_gutenberg_mirror_paths() -> None:
    assert fill.gutenberg_dir(205) == "2/0"
    assert fill.gutenberg_dir(2944) == "2/9/4"


def test_stackexchange_fill_asks_for_full_pages(monkeypatch, tmp_path) -> None:
    listed = [{"id": f"se-{i:012x}", "kind": "stackexchange-answer", "sha256": "x",
               "extraction": corpus.HTML_STRIP_VERSION, "source": {"site": "cooking", "answer_id": i}}
              for i in range(60)]
    _manifest(monkeypatch, tmp_path, listed)
    seen = []

    def get(path, params, _sleep):
        seen.append(params)
        ids = path.split("/")[1].split(";")
        return {"items": [{"answer_id": int(i), "body": f"<p>answer {i}</p>"} for i in ids][: params.get("pagesize", 30)]}

    assert fill.fill_stackexchange(0, get) == (60, 60)
    assert seen == [{"site": "cooking", "pagesize": 100}]
    assert bench.read_text("se-00000000003b") == "answer 59"


def test_human_cut_off_set_skips_edited_sources(monkeypatch, tmp_path) -> None:
    entries = [
        {"id": "cc-000000000001", "kind": "cc-news", "label": "human", "split": "test", "sha256": corpus.sha256_text("same")},
        {"id": "cc-000000000002", "kind": "cc-news", "label": "human", "split": "test", "sha256": corpus.sha256_text("old")},
    ]
    _manifest(monkeypatch, tmp_path, entries)
    corpus.write_cache("cc-000000000001", "same")
    corpus.write_cache("cc-000000000002", "edited")
    assert detect.human_test_texts() == [("cc-000000000001", "same")]
