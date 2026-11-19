# humanizer-pro rewriter benchmark (private)

This repo measures whether the humanizer-pro rewriter (its SKILL.md) makes AI text harder for open-source detectors to spot, without losing facts. It runs only on GitHub Actions, so nothing runs on a personal computer.

It uses only text that anyone can rebuild: the public human pools that humanizer-pro's manifest points to (Wikipedia, Stack Exchange, CC-News, PEPs, Gutenberg, Strunk), the public RAID dataset, and a fresh AI pool generated from humanizer-pro's public prompts. PERSUADE and other private datasets are never built or read here. Corpus text stays in this repo's Actions cache and is never committed. Rewrites, the fresh pool, scores, and reports are committed under `data/`.

## Workflows

All of them are started by hand from the Actions tab.

| Workflow | What it does | Costs money? |
|---|---|---|
| Prepare inputs | Builds the public text and caches it. Each run also fills gaps left by flaky sites (`bench/fill_corpus.py`), so running it again can only add text. With "generate" on, the models answer the 60 public prompts, and the replies are saved to `data/generated/`. | Only with "generate" on |
| Rewrite | One arm (skill, control, or rival) rewrites the fixed sample with each model. The dry run is on by default and only prints the cost. A real run stops before any document that could pass `max_spend`. | Yes, unless dry run |
| Detect | Desklib, Fast-DetectGPT, and Binoculars each score everything on their own runner, then the report is written to the run summary and `data/reports/`. | No (Actions minutes only) |
| Test | Runs the unit tests on every push. | No |

## The numbers in a report

- **Detected:** the share of documents a detector flags. Each detector's cut-off is set so it flags 1% of the real human test documents, so "detected" means the same thing for all three. Lower is better for the rewriter.
- **AUROC:** the chance a detector rates a random document from that row as more AI-like than a random human document. 0.5 means it can't tell them apart, and 1.0 means it always can.
- **Fact drift:** the share of rewrites where compare mode found a lost or changed number, date, name, link, or quote.
- **Meaning warn:** the share where a hedge, negation, or cause changed.
- **Hidden chars:** rewrites containing invisible or lookalike characters. This must stay 0.
- **Median length:** rewrite length divided by the original's.

The skill arm has to beat the control arm (same model, plain "make it sound human" prompt). Otherwise the skill adds nothing beyond the model itself.

## Order of a baseline

1. Prepare inputs with generate on, dry run first, then a real run.
2. Rewrite: skill, then control, then rival, each as a dry run first.
3. Detect with "all".

## Secrets

`OPENROUTER_API_KEY` is the only secret, and it's set under Settings, Secrets and variables, Actions. The workflows pass it only to the steps that call OpenRouter.
