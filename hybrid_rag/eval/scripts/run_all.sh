#!/usr/bin/env bash
# Unified eval runner: ingest, score, and report all three chunking strategies,
# then build the combined comparison -- the whole manual flow in one command.
#
# Each strategy is written to its own index (chunks_<strategy>), so the three
# coexist and no wiping is needed between runs. Outputs:
#
#   hybrid_rag/eval/result/<strategy>/<strategy>.json   raw scores
#   hybrid_rag/eval/result/<strategy>/<strategy>.html   single-strategy report
#   hybrid_rag/eval/result/final/comparison.html        combined comparison
#
# Usage:
#   hybrid_rag/eval/scripts/run_all.sh          # full run over golden_30
#   hybrid_rag/eval/scripts/run_all.sh 5        # cap to 5 questions (cheap smoke)

set -euo pipefail

# Repo root, relative to this script at hybrid_rag/eval/scripts/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

OS_HOST="${OPENSEARCH_HOST:-http://localhost:9200}"
CORPUS="data/eval_test/corpus"
GOLDEN="data/eval_test/golden_30.json"
RESULT_DIR="hybrid_rag/eval/result"
STRATEGIES=(fixed header semantic)

# Optional first arg caps the number of questions (passed to the runner only;
# the full corpus is always ingested so every question's gold docs are present).
LIMIT_ARG=()
if [[ "${1:-}" != "" ]]; then
  LIMIT_ARG=(--limit "$1")
  echo "Question limit: $1"
fi

# Fail early with a clear message if OpenSearch is not up.
if ! curl -sf "$OS_HOST" >/dev/null; then
  echo "ERROR: OpenSearch not reachable at $OS_HOST"
  echo "Start it with: docker compose up -d"
  exit 1
fi

RESULT_FILES=()
for strategy in "${STRATEGIES[@]}"; do
  out_dir="$RESULT_DIR/$strategy"
  mkdir -p "$out_dir"
  echo ""
  echo "=== $strategy ==="
  echo "- ingesting corpus into chunks_$strategy"
  uv run python -m hybrid_rag.eval.ingest --corpus "$CORPUS" --strategy "$strategy"
  echo "- running eval"
  uv run python -m hybrid_rag.eval.runner --strategy "$strategy" \
    --golden "$GOLDEN" --out "$out_dir/$strategy.json" \
    ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}
  echo "- rendering report"
  uv run python -m hybrid_rag.eval.report "$out_dir/$strategy.json" \
    --out "$out_dir/$strategy.html"
  RESULT_FILES+=("$out_dir/$strategy.json")
done

echo ""
echo "=== combined comparison ==="
final_dir="$RESULT_DIR/final"
mkdir -p "$final_dir"
uv run python -m hybrid_rag.eval.compare "${RESULT_FILES[@]}" \
  --out "$final_dir/comparison.html"

echo ""
echo "Done."
echo "Per-strategy: $RESULT_DIR/<strategy>/<strategy>.{json,html}"
echo "Combined:     $final_dir/comparison.html"
