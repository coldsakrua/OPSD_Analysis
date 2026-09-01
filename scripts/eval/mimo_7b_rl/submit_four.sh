#!/bin/bash
# Submit MiMo-7B-RL evals for aime24/25/26 and hmmt25.
# Run from OPSD_Analysis so Slurm --output paths resolve correctly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/mimo-7b-rl}
EVAL_TAG=${EVAL_TAG:-$(basename "${CHECKPOINT_PATH}")}

# Job-name prefix for log/eval/mimo_7b_rl/<ds>/sgl/%x.%j.out
# Baseline: eval_mimo7b_a24 ; trained: st_tt_jsd_clip005_1e6_mimo7b_a24
if [[ "${EVAL_TAG}" == "mimo-7b-rl" || "${EVAL_TAG}" == "checkpoint-100" || "${EVAL_TAG}" == eval_mimo7b* ]]; then
  JOB_PREFIX=eval_mimo7b
else
  JOB_PREFIX="${EVAL_TAG%_ckpt*}"
fi

short_ds() {
  case "$1" in
    aime24) echo a24 ;;
    aime25) echo a25 ;;
    aime26) echo a26 ;;
    hmmt25) echo h25 ;;
    *) echo "$1" ;;
  esac
}

cd "${BASE_DIR}"
mkdir -p log/eval/mimo_7b_rl/{aime24,aime25,aime26,hmmt25}/sgl

echo "[submit] base_dir=${BASE_DIR}"
echo "[submit] checkpoint=${CHECKPOINT_PATH}"
echo "[submit] eval_tag=${EVAL_TAG} job_prefix=${JOB_PREFIX}"
python - "${CHECKPOINT_PATH}" <<'PY'
import json, struct, sys
from pathlib import Path
root = Path(sys.argv[1])
bad = []
shards = sorted(root.glob("model-*.safetensors")) or sorted(root.glob("model.safetensors"))
if not shards:
    raise SystemExit("error: no model*.safetensors in " + str(root))
for p in shards:
    size = p.stat().st_size
    with p.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(header_len))
    ends = [v["data_offsets"][1] for v in meta.values() if isinstance(v, dict) and "data_offsets" in v]
    declared = 8 + header_len + (max(ends) if ends else 0)
    ok = size >= declared
    print(f"[submit] {p.name} size={size} declared={declared} delta={size-declared} ok={ok}")
    if not ok:
        bad.append(p.name)
if bad:
    raise SystemExit("error: truncated safetensors: " + ", ".join(bad))
PY

for ds in aime24 aime25 aime26 hmmt25; do
  job_name="${JOB_PREFIX}_$(short_ds "${ds}")"
  if ((${#job_name} > 48)); then
    job_name="mimo7b_$(short_ds "${ds}")"
  fi
  sbatch \
    --job-name="${job_name}" \
    --output="log/eval/mimo_7b_rl/${ds}/sgl/%x.%j.out" \
    --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}" \
    "${SCRIPT_DIR}/${ds}_sgl.sh"
done
