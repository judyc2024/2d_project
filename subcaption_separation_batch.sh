#!/usr/bin/env bash

# Keep this before `set -u` because some .bashrc files reference unset variables.
source ~/.bashrc
set -euo pipefail

conda activate 2dproject

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OPENI_ROOT="/nfs/turbo/umms-tocho-ns/data/2d_project/oa/unparsed"
OUT_ROOT="/nfs/turbo/umms-tocho-ns/data/2d_project/oa/subfigure_parsed"
PMC_ROOT="/nfs/turbo/umms-tocho-ns/code/chjudy/pmc-data-extraction"

# Either export OPENAI_API_KEY directly, or point OPENAI_ENV_FILE to a file
# containing OPENAI_API_KEY=...
ENV_ARGS=()
if [[ -n "${OPENAI_ENV_FILE:-}" ]]; then
    ENV_ARGS=(--env-file "$OPENAI_ENV_FILE")
fi

MODE="${1:-submit}"

case "$MODE" in
    test)
        # Step 1: Prepare the existing per-folder input JSONLs.
        # Notice that the old --run flag is intentionally absent.
        python "$SCRIPT_DIR/subcaption_separation.py" \
            --openi-root "$OPENI_ROOT" \
            --out-root "$OUT_ROOT" \
            --pmc-root "$PMC_ROOT"

        # Step 2: Submit only one input file = approximately 20 captions.
        python "$SCRIPT_DIR/subcaption_batch_api.py" submit \
            --out-root "$OUT_ROOT" \
            --pmc-root "$PMC_ROOT" \
            --model "gpt-4o-mini" \
            --max-tokens 500 \
            --max-input-files 1 \
            --max-requests-per-batch 20 \
            "${ENV_ARGS[@]}"
        ;;

    submit)
        # Prepare all per-folder input JSONLs.
        python "$SCRIPT_DIR/subcaption_separation.py" \
            --openi-root "$OPENI_ROOT" \
            --out-root "$OUT_ROOT" \
            --pmc-root "$PMC_ROOT"

        # Combine them into larger OpenAI batches.
        python "$SCRIPT_DIR/subcaption_batch_api.py" submit \
            --out-root "$OUT_ROOT" \
            --pmc-root "$PMC_ROOT" \
            --model "gpt-4o-mini" \
            --max-tokens 500 \
            --max-requests-per-batch 10000 \
            "${ENV_ARGS[@]}"
        ;;

    status)
        python "$SCRIPT_DIR/subcaption_batch_api.py" status \
            --out-root "$OUT_ROOT" \
            --pmc-root "$PMC_ROOT" \
            "${ENV_ARGS[@]}"
        ;;

    collect)
        python "$SCRIPT_DIR/subcaption_batch_api.py" collect \
            --out-root "$OUT_ROOT" \
            --pmc-root "$PMC_ROOT" \
            "${ENV_ARGS[@]}"
        ;;

    *)
        echo "Usage: $0 {test|submit|status|collect}" >&2
        exit 2
        ;;
esac
