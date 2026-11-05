#!/bin/bash
# Submit 1-GPU LCB index-slice shards (val-n=4, seed=42), then merge+judge.
#
# Usage (resume after first 54 problems already generated):
#   CHECKPOINT_PATH=... EVAL_TAG=... START_FROM=54 CHUNK=24 \
#     BASE_JSONL=.../lcb_v6_olmo3_7b_think_sgl_think.generations.base0_54.jsonl \
#     bash scripts/eval_code/olmo3-7b-think/submit_lcb_v6_think_index_shards.sh
#
# Optional: END_AT=175 EXTRA_RANGES="..." AUTO_JUDGE=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/olmo-3-7b-think
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${DEFAULT_CKPT}}
EVAL_TAG=${EVAL_TAG:-$(basename "${CHECKPOINT_PATH}")}
DATASET=lcb_v6
MODEL_TAG=olmo3_7b_think
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_sgl_think.json}
DATA_JSONL=${DATA_JSONL:-${BASE_DIR}/data/livecodebench/test6.jsonl}
SEED=${SEED:-42}
VAL_N=${VAL_N:-4}
GENERATE_BATCH_SIZE=${GENERATE_BATCH_SIZE:-6}
START_FROM=${START_FROM:-0}
END_AT=${END_AT:-}
CHUNK=${CHUNK:-24}
EXTRA_RANGES=${EXTRA_RANGES:-}
BASE_JSONL=${BASE_JSONL:-}
AUTO_JUDGE=${AUTO_JUDGE:-1}
WORKER="${BASE_DIR}/scripts/eval_code/olmo3-7b-think/lcb_v6_think.sh"
MERGE="${BASE_DIR}/scripts/eval_code/olmo3-7b-think/lcb_v6_think_merge_index_judge.sh"

if [[ -z "${END_AT}" ]]; then
  END_AT=$(wc -l < "${DATA_JSONL}")
fi

name_prefix="${EVAL_TAG}"
name_prefix="${name_prefix%_checkpoint-100}"
name_prefix="${name_prefix%_ckpt100}"
name_prefix="${name_prefix:0:40}"

cd "${BASE_DIR}"
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval_code/olmo3-7b-think/lcb_v6/sgl"

ranges=()
if [[ -n "${EXTRA_RANGES}" ]]; then
  IFS=',' read -r -a extra <<< "${EXTRA_RANGES}"
  for r in "${extra[@]}"; do
    r="${r// /}"
    [[ -n "${r}" ]] && ranges+=("${r}")
  done
fi
for ((s = START_FROM; s < END_AT; s += CHUNK)); do
  e=$((s + CHUNK))
  if ((e > END_AT)); then
    e=${END_AT}
  fi
  ranges+=("${s}:${e}")
done

echo "[submit] checkpoint=${CHECKPOINT_PATH}"
echo "[submit] eval_tag=${EVAL_TAG}"
echo "[submit] final_output=${OUTPUT_JSON}"
echo "[submit] base_jsonl=${BASE_JSONL:-<none>}"
echo "[submit] ranges=${ranges[*]} seed=${SEED} val_n=${VAL_N} chunk=${CHUNK}"

dep_ids=()
shard_list_file="$(dirname "${OUTPUT_JSON}")/.$(basename "${OUTPUT_JSON%.json}").shard_ranges.txt"
: > "${shard_list_file}"
for range in "${ranges[@]}"; do
  start="${range%%:*}"
  end="${range##*:}"
  shard_out=$(printf "%s.i%03d_%03d.json" "${OUTPUT_JSON%.json}" "${start}" "${end}")
  rm -f "${shard_out}" "${shard_out%.json}.generations.json" "${shard_out%.json}.generations.jsonl"
  jname=$(printf "%s_i%03d_%03d" "${name_prefix}" "${start}" "${end}")
  jname="${jname:0:64}"
  out=$(
    sbatch --parsable \
      --job-name="${jname}" \
      --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}",OUTPUT_JSON="${shard_out}",DATA_JSONL="${DATA_JSONL}",SEED="${SEED}",VAL_N="${VAL_N}",AUTO_JUDGE=0,GENERATE_BATCH_SIZE="${GENERATE_BATCH_SIZE}",START_IDX="${start}",END_IDX="${end}" \
      --chdir="${BASE_DIR}" \
      "${WORKER}"
  )
  jid="${out%%;*}"
  dep_ids+=("${jid}")
  echo "${start}:${end}" >> "${shard_list_file}"
  echo "[submit] shard [${start}:${end}) job=${jid} out=${shard_out}"
done
echo "[submit] shard_list_file=${shard_list_file}"

dep_csv=$(IFS=:; echo "${dep_ids[*]}")
merge_name="${name_prefix}_idx_merge"
merge_name="${merge_name:0:64}"
mout=$(
  sbatch --parsable \
    --job-name="${merge_name}" \
    --dependency=afterok:${dep_csv} \
    --export=ALL,BASE_DIR="${BASE_DIR}",OUTPUT_JSON="${OUTPUT_JSON}",DATA_JSONL="${DATA_JSONL}",EVAL_TAG="${EVAL_TAG}",BASE_JSONL="${BASE_JSONL}",SHARD_LIST_FILE="${shard_list_file}",VAL_N="${VAL_N}",AUTO_JUDGE="${AUTO_JUDGE}" \
    --chdir="${BASE_DIR}" \
    "${MERGE}"
)
mjid="${mout%%;*}"
echo "[submit] merge+judge job=${mjid} afterok:${dep_csv}"
echo "[submit] done"
