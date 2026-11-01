#!/bin/bash
# Submit 4 independent 1-GPU LCB jobs (val-n=1 each, different seeds), then
# one merge+judge job after all succeed.
#
# Usage:
#   CHECKPOINT_PATH=... EVAL_TAG=... bash scripts/eval_code/falcon_h1r_7b/submit_lcb_v6_sgl_4x1gpu.sh
# Optional: BASE_SEED=42 N1_SHARDS=4 GENERATE_BATCH_SIZE=2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/falcon-h1r-7b
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${DEFAULT_CKPT}}
EVAL_TAG=${EVAL_TAG:-$(basename "${CHECKPOINT_PATH}")}
DATASET=lcb_v6
MODEL_TAG=falcon_h1r_7b
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_sgl.json}
DATA_JSONL=${DATA_JSONL:-${BASE_DIR}/data/livecodebench/test6.jsonl}
BASE_SEED=${BASE_SEED:-42}
N1_SHARDS=${N1_SHARDS:-4}
GENERATE_BATCH_SIZE=${GENERATE_BATCH_SIZE:-2}
WORKER="${BASE_DIR}/scripts/eval_code/falcon_h1r_7b/lcb_v6_sgl.sh"
MERGE="${BASE_DIR}/scripts/eval_code/falcon_h1r_7b/lcb_v6_sgl_merge_judge.sh"

# Short job-name prefix (Slurm limit); keep tag recognizable.
name_prefix="${EVAL_TAG}"
name_prefix="${name_prefix%_checkpoint-100}"
name_prefix="${name_prefix%_ckpt100}"
name_prefix="${name_prefix:0:40}"

cd "${BASE_DIR}"
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval_code/falcon_h1r_7b/lcb_v6/sgl"

# Avoid resuming into previous tp4 garbage under the same final path; shards are separate.
echo "[submit] checkpoint=${CHECKPOINT_PATH}"
echo "[submit] eval_tag=${EVAL_TAG}"
echo "[submit] final_output=${OUTPUT_JSON}"
echo "[submit] n1_shards=${N1_SHARDS} base_seed=${BASE_SEED} batch=${GENERATE_BATCH_SIZE}"

dep_ids=()
for i in $(seq 0 $((N1_SHARDS - 1))); do
  seed=$((BASE_SEED + i))
  shard_out="${OUTPUT_JSON%.json}.n1_s${i}.json"
  # Fresh shard (do not resume old partial / wrong prompt leftovers).
  rm -f "${shard_out}" "${shard_out%.json}.generations.json" "${shard_out%.json}.generations.jsonl"
  jname="${name_prefix}_lcb_s${i}"
  jname="${jname:0:64}"
  out=$(
    sbatch --parsable \
      --job-name="${jname}" \
      --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}",OUTPUT_JSON="${shard_out}",DATA_JSONL="${DATA_JSONL}",SEED="${seed}",VAL_N=1,PASS_AT_K=1,AUTO_JUDGE=0,GENERATE_BATCH_SIZE="${GENERATE_BATCH_SIZE}",SAMPLE_IDX="${i}" \
      --chdir="${BASE_DIR}" \
      "${WORKER}"
  )
  jid="${out%%;*}"
  dep_ids+=("${jid}")
  echo "[submit] shard${i} job=${jid} seed=${seed} out=${shard_out}"
done

dep_csv=$(IFS=:; echo "${dep_ids[*]}")
merge_name="${name_prefix}_lcb_merge"
merge_name="${merge_name:0:64}"
mout=$(
  sbatch --parsable \
    --job-name="${merge_name}" \
    --dependency=afterok:${dep_csv} \
    --export=ALL,BASE_DIR="${BASE_DIR}",OUTPUT_JSON="${OUTPUT_JSON}",DATA_JSONL="${DATA_JSONL}",EVAL_TAG="${EVAL_TAG}",N1_SHARDS="${N1_SHARDS}",AUTO_JUDGE=1 \
    --chdir="${BASE_DIR}" \
    "${MERGE}"
)
mjid="${mout%%;*}"
echo "[submit] merge+judge job=${mjid} afterok:${dep_csv}"
echo "[submit] done"
