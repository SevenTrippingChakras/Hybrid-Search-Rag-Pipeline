# Evaluation

Scores the RAG pipeline against a golden question set and compares chunking
strategies. Five metrics per question: faithfulness and answer correctness
(RAGAS, LLM-judged) plus retrieval relevance, citation accuracy, and abstention
accuracy (computed locally, no LLM). See
[`docs/evaluation-framework.md`](../../docs/evaluation-framework.md) for what each
metric means and how a "best strategy" verdict is reached.

## Modules

- `ingest.py` - fills a per-strategy OpenSearch index (`chunks_<strategy>`)
- `metrics.py` - the three local, no-LLM metrics
- `ragas_scorer.py` - faithfulness + answer correctness via RAGAS
- `runner.py` - runs the pipeline over the golden set, writes `results.json`
- `report.py` - single-run HTML report
- `compare.py` - combined chunking-strategy comparison HTML
- `results.py` - the `results.json` schema (shared by runner and reports)
- `judge.py` - the OpenAI judge wiring RAGAS uses

## Prerequisites

```bash
cd "<repo root>"

# OpenAI key the judge and embeddings use (read from .env)
echo "OPENAI_API_KEY=sk-..." >> .env

docker compose up -d            # start local OpenSearch
```

The unified script (below) creates its own output folders. For the manual steps,
outputs go under `app/rag/eval/result/`.

Each strategy is written to its **own** OpenSearch index (`chunks_fixed`,
`chunks_header`, `chunks_semantic`), so the three coexist and no wiping is needed
between runs. Ingest writes to `chunks_<strategy>`; the runner points retrieval at
the matching index.

## Run everything (one command)

`scripts/run_all.sh` does the whole flow for all three strategies -- ingest,
eval, report -- then builds the combined comparison:

```bash
app/rag/eval/scripts/run_all.sh          # full run over golden_30
app/rag/eval/scripts/run_all.sh 5        # cap to 5 questions (cheap smoke)
```

Outputs are written under `app/rag/eval/result/`:

```
app/rag/eval/result/
  fixed/     fixed.json     fixed.html
  header/    header.json    header.html
  semantic/  semantic.json  semantic.html
  final/     comparison.html      <- the combined comparison report
```

The optional first argument caps the number of questions (passed to the runner
only; the full corpus is always ingested so every question's gold docs are
present). Override the OpenSearch endpoint with `OPENSEARCH_HOST` if not on the
local default.

The sections below document the individual steps the script runs, for when you
want to run one strategy by hand.

## Run one strategy

Replace `STRATEGY` with `fixed`, `header`, or `semantic`:

```bash
mkdir -p app/rag/eval/result/STRATEGY
uv run python -m app.rag.eval.ingest --corpus data/eval_test/corpus --strategy STRATEGY
uv run python -m app.rag.eval.runner --strategy STRATEGY \
  --out app/rag/eval/result/STRATEGY/STRATEGY.json
uv run python -m app.rag.eval.report app/rag/eval/result/STRATEGY/STRATEGY.json \
  --out app/rag/eval/result/STRATEGY/STRATEGY.html
```

Repeat for all three strategies -- each writes to its own `chunks_STRATEGY` index.

## Combined comparison report

Once all three strategy result files exist:

```bash
mkdir -p app/rag/eval/result/final
uv run python -m app.rag.eval.compare \
  app/rag/eval/result/fixed/fixed.json \
  app/rag/eval/result/header/header.json \
  app/rag/eval/result/semantic/semantic.json \
  --out app/rag/eval/result/final/comparison.html
open app/rag/eval/result/final/comparison.html
```

## Options

- `runner` defaults `--golden` to `data/eval_test/golden_30.json`; pass `--golden`
  to point at a different set.
- Add `--limit N` to `runner` (and to `ingest`) to cap questions/docs for a cheap
  first run while watching cost.
- The full 3-strategy run on the 300-doc / 30-question eval set costs well under
  a few dollars with `gpt-4o-mini`; the RAGAS judge is the only meaningful spend.

## Smoke test (optional, near-zero cost)

Confirm the wiring before a full run:

```bash
uv run python -m app.rag.eval.ingest --corpus data/eval_test/corpus --strategy fixed --limit 3
uv run python -m app.rag.eval.runner --strategy fixed --out /tmp/smoke.json --limit 3
```

Or just run the unified script with a small cap: `scripts/run_all.sh 5`.
