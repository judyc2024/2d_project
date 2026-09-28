#!/usr/bin/env bash

source ~/.bashrc
conda activate 2dproject

set -uo pipefail

# Activate the environment when necessary.
# source /nfs/turbo/umms-tocho-ns/code/chjudy/pmc-data-extraction/.env/bin/activate

input_root="/nfs/turbo/umms-tocho-ns/data/2d_project/oa/unparsed"
output_root="/nfs/turbo/umms-tocho-ns/data/2d_project/oa/subfigure_parsed"
log_dir="${output_root}/parallel_logs"

mkdir -p "${log_dir}"

mapfile -t journals < <(cd "${input_root}" && ls -d */ | tr -d '/')

pids=()
groups=()

# I am submitting jobs in parallel without one job to finish 
for gpu in 0 1 2 3; do
    # Deal the journals round-robin so each GPU gets a similar share.
    group=()
    for i in "${!journals[@]}"; do
        if (( i % 4 == gpu )); then
            group+=("${journals[$i]}")
        fi
    done

    if [[ "${#group[@]}" -eq 0 ]]; then
        continue
    fi

    log_file="${log_dir}/gpu${gpu}.log"

    echo "Launching ${group[*]} on GPU ${gpu}"
    echo "Log: ${log_file}"

    # CUDA_VISIBLE_DEVICES="${gpu}" \
    python subfigure_separation.py \
        --input-root "${input_root}" \
        --output-root "${output_root}" \
        --journals "${group[@]}" \
        --run \
        --continue-on-error \
        --skip-existing \
        --batch-size 8 \
        --num-workers 2 \
        --gpu "${gpu}" \
        > "${log_file}" 2>&1 &

    pids+=("$!")
    groups+=("${group[*]}")
done

failed=0

for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    group="${groups[$i]}"

    if wait "${pid}"; then
        echo "Completed successfully: ${group}"
    else
        status=$?
        echo "Failed: ${group}, exit code ${status}" >&2
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    echo "One or more journal jobs failed. Check ${log_dir}." >&2
    exit 1
fi

echo "All journals completed successfully."