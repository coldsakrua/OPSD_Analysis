#!/bin/bash
# After seed-42 first256 training finishes, eval at seeds 42 and 0-6
# (plus 1024/65536 to match the 10-seed set used elsewhere).
#
# Usage:
#   TRAIN_JOBID=<id> bash submit_olmo_first256_seed42_eval_after_train.sh
# Or with dependency already set externally:
#   CHECKPOINT_PATH=... bash submit_olmo_first256_seed42_eval_after_train.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

RUN_NAME=st_tt_clip005_c1024_first256_olmo7bt
if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  : "${TRAIN_JOBID:?Set TRAIN_JOBID or CHECKPOINT_PATH}"
  CHECKPOINT_PATH="${ROOT}/outputs/olmo3_7b_think/${RUN_NAME}/${TRAIN_JOBID}/checkpoint-100"
fi
if [[ ! -f "${CHECKPOINT_PATH}/config.json" ]]; then
  echo "[error] missing checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

TAG_BASE="${RUN_NAME}_checkpoint-100"
for seed in 42 0 1 2 3 4 5 6 1024 65536; do
  if [[ "${seed}" == "42" ]]; then
    tag="${TAG_BASE}"
  else
    tag="${TAG_BASE}_seed${seed}"
  fi
  out_dir="${ROOT}/eval_outputs/${tag}"
  n_ok=$(ls "${out_dir}"/aime24*.metrics.json "${out_dir}"/aime25*.metrics.json \
               "${out_dir}"/aime26*.metrics.json "${out_dir}"/hmmt25*.metrics.json \
               2>/dev/null | wc -l || true)
  if [[ "${n_ok}" -ge 4 ]]; then
    echo "[olmo-first256-s42-eval] skip seed=${seed} (already have 4 metrics)"
    continue
  fi
  echo "[olmo-first256-s42-eval] seed=${seed} tag=${tag}"
  CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    EVAL_TAG="${tag}" \
    SEED="${seed}" \
    BASE_DIR="${ROOT}" \
    bash "${ROOT}/scripts/eval/olmo_7b_think/seed/submit_four_seed.sh"
done
echo "[olmo-first256-s42-eval] done"
