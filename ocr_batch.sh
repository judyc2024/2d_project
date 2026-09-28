#!/usr/bin/env bash

# Keep this before `set -u` because some .bashrc files reference unset variables.
source ~/.bashrc
set -euo pipefail

conda activate 2dproject

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="/nfs/turbo/umms-tocho-ns/data/2d_project/oa/subfigure_parsed"

# Either export OPENAI_API_KEY directly, or point OPENAI_ENV_FILE to a file
# containing OPENAI_API_KEY=...
ENV_ARGS=()
if [[ -n "${OPENAI_ENV_FILE:-}" ]]; then
    ENV_ARGS=(--env-file "$OPENAI_ENV_FILE")
fi

MODE="${1:-submit}"

case "$MODE" in
    test)
        # Small smoke test: one input jsonl, tiny batch.
        python "$SCRIPT_DIR/ocr.py" submit \
            --data-root "$DATA_ROOT" \
            --model "gpt-5.4-mini" \
            --max-input-files 1 \
            --max-requests-per-batch 20 \
            --max-batch-mb 190 \
            --skip-existing \
            "${ENV_ARGS[@]}"
        ;;

    submit)
        # Pack as many image requests as fit under the OpenAI 200 MB batch
        # file limit (~190 MB with headroom). Request count alone is a poor
        # limiter here because base64 images dominate file size.
        python "$SCRIPT_DIR/ocr.py" submit \
            --data-root "$DATA_ROOT" \
            --model "gpt-5.4-mini" \
            --max-requests-per-batch 50000 \
            --max-batch-mb 190 \
            --skip-existing \
            "${ENV_ARGS[@]}"
        ;;

    status)
        python "$SCRIPT_DIR/ocr.py" status \
            --data-root "$DATA_ROOT" \
            "${ENV_ARGS[@]}"
        ;;

    collect)
        python "$SCRIPT_DIR/ocr.py" collect \
            --data-root "$DATA_ROOT" \
            "${ENV_ARGS[@]}"
        ;;

    *)
        echo "Usage: $0 {test|submit|status|collect}" >&2
        exit 2
        ;;
esac
