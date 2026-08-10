#!/bin/bash
# Watch snt_tnt LoRA train jobs; after checkpoint-100 exists, submit evals.
# - olmo 3212531 → merge → aime24/25/26/hmmt25_sgl.sh
# - 4b  3212546 → lora /*_nothink.sh (vLLM adapter)
# - q35 3212547 → merge → lora /*_nothink.sh (SGLang merged)
set -euo pipefail

BASE_DIR=/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis
cd "${BASE_DIR}"
STATE_DIR="${BASE_DIR}/log/train/_watch_snt_tnt_lora"
mkdir -p "${STATE_DIR}"
LOG="${STATE_DIR}/watch.$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

declare -A DONE=()
[[ -f "${STATE_DIR}/done.4b" ]] && DONE[4b]=1
[[ -f "${STATE_DIR}/done.q35" ]] && DONE[q35]=1
[[ -f "${STATE_DIR}/done.olmo" ]] && DONE[olmo]=1

job_done() {
  local jid="$1"
  local st
  st=$(sacct -j "${jid}" -n -X -o State -P 2>/dev/null | head -1 | tr -d ' ')
  case "${st}" in
    COMPLETED) return 0 ;;
    FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED) echo "[watch] job ${jid} ended badly: ${st}"; return 2 ;;
    *) return 1 ;;
  esac
}

ckpt_ready() {
  local d="$1"
  [[ -d "${d}/checkpoint-100" ]] || return 1
  # adapter or full weights
  [[ -f "${d}/checkpoint-100/adapter_config.json" || -f "${d}/checkpoint-100/config.json" ]]
}

submit_4b() {
  local ckpt="$1"
  local tag="$2"
  local ds j
  echo "[watch] submit 4b LoRA nothink evals ckpt=${ckpt} tag=${tag}"
  for ds in aime24 aime25 aime26 hmmt25; do
    local abbr
    case "${ds}" in aime24) abbr=a24;; aime25) abbr=a25;; aime26) abbr=a26;; hmmt25) abbr=h25;; esac
    j=$(CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${tag}" sbatch --parsable \
      --job-name="snt_tnt_lora_clip005_lr5e6_4b_${abbr}_nt" \
      "scripts/eval/4b/lora/${ds}_nothink.sh")
    echo "  ${j}  ${ds}_nothink"
  done
  touch "${STATE_DIR}/done.4b"
}

submit_merge_then_sgl_olmo() {
  local adapter="$1"
  local merged="$2"
  local tag="$3"
  local merge_j j ds
  echo "[watch] submit olmo LoRA merge → ${merged}"
  mkdir -p "$(dirname "${merged}")" log/eval/olmo3_7b_instruct/lora/merge
  merge_j=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=merge_snt_tnt_lora_olmo7bi_ckpt100
#SBATCH --output=log/eval/olmo3_7b_instruct/lora/merge/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2:00:00
set -euo pipefail
cd "${BASE_DIR}"
set +u; source activate sglang; set -u
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
python - <<'PY'
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
BASE = Path("/gpfs/share/home/2501210611/labShare/2501210611/model/olmo3-7b-it")
adapter = Path("${adapter}")
out = Path("${merged}")
if out.exists() and (out / "config.json").exists() and list(out.glob("*.safetensors")):
    print("[merge] skip existing", out, flush=True)
else:
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(BASE), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE), torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cuda", low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    n = sum(1 for n,_ in model.named_parameters() if "lora_" in n)
    print(f"[merge] lora params={n}", flush=True)
    if n < 50:
        raise SystemExit(f"too few lora params: {n}")
    model = model.merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    print("[merge] done", out, flush=True)
PY
EOF
)
  echo "[watch] olmo merge job=${merge_j}"
  for ds in aime24 aime25 aime26 hmmt25; do
    local abbr
    case "${ds}" in aime24) abbr=a24;; aime25) abbr=a25;; aime26) abbr=a26;; hmmt25) abbr=h25;; esac
    j=$(CHECKPOINT_PATH="${merged}" EVAL_TAG="${tag}" sbatch --parsable \
      --dependency=afterok:${merge_j} \
      --job-name="snt_tnt_lora_clip005_lr5e6_olmo7bi_${abbr}_sgl" \
      "scripts/eval/olmo3_7b_instruct/${ds}_sgl.sh")
    echo "  ${j}  ${ds}_sgl (dep=${merge_j})"
  done
  touch "${STATE_DIR}/done.olmo"
}

submit_merge_then_q35() {
  local adapter="$1"
  local merged="$2"
  local tag="$3"
  local merge_j j ds
  echo "[watch] submit q35 LoRA merge → ${merged}"
  mkdir -p "$(dirname "${merged}")" log/eval/qwen3.5_4b/lora/merge
  merge_j=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=merge_snt_tnt_lora_q35_ckpt100
#SBATCH --output=log/eval/qwen3.5_4b/lora/merge/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2:00:00
set -euo pipefail
cd "${BASE_DIR}"
set +u; source activate qwen3_5; set -u
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
python - <<'PY'
import shutil
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
BASE = Path("/gpfs/share/home/2501210611/labShare/2501210611/model/qwen35_4b")
ASSETS = (
    "preprocessor_config.json", "video_preprocessor_config.json",
    "merges.txt", "vocab.json", "chat_template.jinja", "configuration.json",
)
adapter = Path("${adapter}")
out = Path("${merged}")
if out.exists() and (out / "config.json").exists() and list(out.glob("*.safetensors")):
    print("[merge] skip existing", out, flush=True)
else:
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(BASE), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(BASE), dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cuda", low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    n = sum(1 for n,_ in model.named_parameters() if "lora_" in n)
    print(f"[merge] lora params={n}", flush=True)
    if n < 100:
        raise SystemExit(f"too few lora params: {n}")
    model = model.merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    for name in ASSETS:
        src, dst = BASE / name, out / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
    print("[merge] done", out, flush=True)
PY
EOF
)
  echo "[watch] q35 merge job=${merge_j}"
  for ds in aime24 aime25 aime26 hmmt25; do
    local abbr
    case "${ds}" in aime24) abbr=a24;; aime25) abbr=a25;; aime26) abbr=a26;; hmmt25) abbr=h25;; esac
    j=$(CHECKPOINT_PATH="${merged}" EVAL_TAG="${tag}" sbatch --parsable \
      --dependency=afterok:${merge_j} \
      --job-name="snt_tnt_lora_clip005_lr5e6_q35_4b_${abbr}_nt" \
      "scripts/eval/qwen3.5_4b/lora/${ds}_nothink.sh")
    echo "  ${j}  ${ds}_nothink (dep=${merge_j})"
  done
  touch "${STATE_DIR}/done.q35"
}

echo "[watch] started $(date) log=${LOG}"
echo "[watch] tracking olmo=3212531 4b=3212546 q35=3212547"

while true; do
  # ---- 4b ----
  if [[ -z "${DONE[4b]:-}" ]]; then
    if ckpt_ready "outputs/qwen3_4b/snt_tnt_lora_clip005_lr5e6_openthoughts_4b/3212546"; then
      rc=0; job_done 3212546 || rc=$?
      if [[ ${rc} -eq 0 ]]; then
        submit_4b \
          "outputs/qwen3_4b/snt_tnt_lora_clip005_lr5e6_openthoughts_4b/3212546/checkpoint-100" \
          "snt_tnt_lora_clip005_lr5e6_4b_ckpt100"
        DONE[4b]=1
      elif [[ ${rc} -eq 2 ]]; then DONE[4b]=1; fi
    fi
  fi

  # ---- q35 ----
  if [[ -z "${DONE[q35]:-}" ]]; then
    if ckpt_ready "outputs/qwen3.5_4b/snt_tnt_lora_clip005_lr5e6_openthoughts_q35_4b/3212547"; then
      rc=0; job_done 3212547 || rc=$?
      if [[ ${rc} -eq 0 ]]; then
        submit_merge_then_q35 \
          "outputs/qwen3.5_4b/snt_tnt_lora_clip005_lr5e6_openthoughts_q35_4b/3212547/checkpoint-100" \
          "outputs/qwen3.5_4b/snt_tnt_lora_clip005_lr5e6_openthoughts_q35_4b/3212547/merged_ckpt100" \
          "snt_tnt_lora_clip005_lr5e6_q35_4b_ckpt100"
        DONE[q35]=1
      elif [[ ${rc} -eq 2 ]]; then DONE[q35]=1; fi
    fi
  fi

  # ---- olmo ----
  if [[ -z "${DONE[olmo]:-}" ]]; then
    if ckpt_ready "outputs/olmo3_7b_instruct/snt_tnt_lora_clip005_lr5e6_openthoughts_olmo7bit/3212531"; then
      rc=0; job_done 3212531 || rc=$?
      if [[ ${rc} -eq 0 ]]; then
        submit_merge_then_sgl_olmo \
          "outputs/olmo3_7b_instruct/snt_tnt_lora_clip005_lr5e6_openthoughts_olmo7bit/3212531/checkpoint-100" \
          "outputs/olmo3_7b_instruct/snt_tnt_lora_clip005_lr5e6_openthoughts_olmo7bit/3212531/merged_ckpt100" \
          "snt_tnt_lora_clip005_lr5e6_olmo7bit_ckpt100"
        DONE[olmo]=1
      elif [[ ${rc} -eq 2 ]]; then DONE[olmo]=1; fi
    fi
  fi

  if [[ -n "${DONE[4b]:-}" && -n "${DONE[q35]:-}" && -n "${DONE[olmo]:-}" ]]; then
    echo "[watch] all three handled $(date)"
    echo 'AGENT_LOOP_WAKE_snt_tnt_lora_eval {"prompt":"snt_tnt LoRA train watch finished; summarize submitted eval jobs"}'
    exit 0
  fi

  # progress line
  for jid in 3212531 3212546 3212547; do
    st=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || sacct -j "${jid}" -n -X -o State -P 2>/dev/null | head -1 | tr -d ' ')
    echo "[watch] $(date +%H:%M:%S) job ${jid} state=${st:-unknown}"
  done
  sleep 300
done
