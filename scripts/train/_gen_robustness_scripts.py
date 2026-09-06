#!/usr/bin/env python3
"""One-shot generator for robustness train/eval scripts. Safe to re-run."""
from __future__ import annotations

import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"[write] {path.relative_to(ROOT)}")


def sbatch_header_1p7(job: str, time: str = "3:00:00") -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --output=log/train/1.7b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time={time}
#SBATCH --exclude=gpua800n13
set -euo pipefail
"""


def sbatch_header_olmo(job: str, time: str = "03:00:00") -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --output=log/train/olmo3-7b-think/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time={time}
#SBATCH --exclude=gpua800n03,gpua800n10,gpua800n13,gpua800n21
set -euo pipefail
"""


def make_olmo_token_select(olmo_base: str, kind: str, tokens: int = 256) -> str:
    var_cli = {
        "first": ("FIRST_LOSS_TOKENS", "--first-loss-tokens"),
        "uni": ("UNIFORM_LOSS_TOKENS", "--uniform-loss-tokens"),
        "last": ("LAST_LOSS_TOKENS", "--last-loss-tokens"),
    }[kind]
    var, cli = var_cli
    job = f"st_tt_clip005_c1024_{kind}{tokens}_olmo7bt"
    run = f"st_tt_clip005_c1024_{kind}{tokens}_olmo7bt"
    text = olmo_base
    text = text.replace("st_tt_clip005_1e6_olmo7bt", job, 1)
    text = text.replace(
        "RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_openthoughts_olmo7bt}",
        f"RUN_NAME=${{RUN_NAME:-{run}}}",
    )
    needle = "MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-1024}\n"
    insert = (
        "MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-1024}\n"
        f"{var}=${{{var}:-{tokens}}}\n"
    )
    if needle in text:
        text = text.replace(needle, insert, 1)
    old = '  --seed "${SEED}" \\\n  "${THINK_ARGS[@]}"'
    new = (
        f'  {cli} "${{{var}}}" \\\n'
        '  --seed "${SEED}" \\\n'
        '  "${THINK_ARGS[@]}"'
    )
    text = text.replace(old, new)
    header = (
        f"# c1024 + {kind}{tokens} variant (Olmo-3-7B-Think)\n"
        f"#   c1024: max_completion=1024\n"
        f"#   {kind}{tokens}: {cli} {tokens}\n"
    )
    text = text.replace("set -euo pipefail\n\n", f"set -euo pipefail\n\n{header}", 1)
    text = text.replace(
        'echo "[launch] lr=${LEARNING_RATE} jsd_token_clip=${JSD_TOKEN_CLIP} seed=${SEED}"',
        'echo "[launch] lr=${LEARNING_RATE} jsd_token_clip=${JSD_TOKEN_CLIP} '
        f'seed=${{SEED}} {kind}_loss_tokens=${{{var}}}"',
    )
    return text


def make_seed_train_wrapper(header: str, parent_rel: str, job: str, run: str, seed: int, comment: str) -> str:
    parent_abs = ROOT / parent_rel
    return (
        header
        + f"""
# Multi-seed train wrapper (seed={seed}). Same hyperparams as parent; only SEED/RUN_NAME differ.
# Parent: {parent_rel}
# Eval these ckpts with default SEED=42.
{comment}
export SEED={seed}
export RUN_NAME={run}
export BASE_DIR="${{BASE_DIR:-${{SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}}}"
bash "{parent_abs}"
"""
    )


def main() -> None:
    # ----- first256 for 1.7b (from last256) -----
    last256_1p7 = (
        ROOT / "scripts/train/qwen3_1.7b/jsd005/opsd_st_tt_clip005_c1024_last256_openthoughts.sh"
    ).read_text()
    first256_1p7 = last256_1p7
    first256_1p7 = first256_1p7.replace("last256", "first256")
    first256_1p7 = first256_1p7.replace("LAST_LOSS_TOKENS", "FIRST_LOSS_TOKENS")
    first256_1p7 = first256_1p7.replace("--last-loss-tokens", "--first-loss-tokens")
    first256_1p7 = first256_1p7.replace("last 256", "first 256")
    first256_1p7 = first256_1p7.replace("(last)", "(first)")
    write(
        ROOT / "scripts/train/qwen3_1.7b/jsd005/opsd_st_tt_clip005_c1024_first256_openthoughts.sh",
        first256_1p7,
    )

    olmo_base = (
        ROOT / "scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts.sh"
    ).read_text()
    for kind in ["first", "uni", "last"]:
        write(
            ROOT / f"scripts/train/olmo3_7b_think/jsd005/opsd_st_tt_clip005_c1024_{kind}256_openthoughts.sh",
            make_olmo_token_select(olmo_base, kind),
        )

    # ----- topk 1.7b -----
    ref_1p7 = (
        ROOT / "scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts.sh"
    ).read_text()
    for k in [1, 4, 16]:
        text = ref_1p7
        job = f"st_tt_clip005_topk{k}_1p7b"
        run = f"st_tt_clip005_topk{k}_c1024_1p7b"
        text = text.replace("st_tt_clip005_1e6_1p7b", job, 1)
        text = text.replace(
            "RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_openthoughts_1p7b}",
            f"RUN_NAME=${{RUN_NAME:-{run}}}",
        )
        text = text.replace(
            "export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_1p7b_fullparam_100step_openthoughts}",
            f"export WANDB_RUN_GROUP=${{WANDB_RUN_GROUP:-qwen3_1p7b_topk{k}_openthoughts}}",
        )
        text = text.replace(
            '  --seed "${SEED}" \\\n  "${THINK_ARGS[@]}"',
            '  --top-k-loss ' + str(k) + ' \\\n  --seed "${SEED}" \\\n  "${THINK_ARGS[@]}"',
        )
        note = (
            f"# Top-k KL (k={k}); k=1 → log π_T(x)-log π_S(x); k>1 → teacher top-k renorm forward KL.\n"
            "# Rollout c1024; jsd_token_clip=0.05; default beta=0 (forward KL).\n"
        )
        text = text.replace("set -euo pipefail\n\n", f"set -euo pipefail\n\n{note}", 1)
        text = text.replace(
            'echo "[launch] lr=${LEARNING_RATE} jsd_token_clip=${JSD_TOKEN_CLIP} seed=${SEED}"',
            'echo "[launch] lr=${LEARNING_RATE} jsd_token_clip=${JSD_TOKEN_CLIP} '
            f'seed=${{SEED}} top_k_loss={k}"',
        )
        write(ROOT / f"scripts/train/qwen3_1.7b/topk/opsd_st_tt_clip005_topk{k}_c1024_ot.sh", text)

    # ----- topk olmo (reverse KL: beta=1, student TopK) -----
    for k in [1, 4, 16]:
        text = olmo_base
        job = f"st_tt_clip005_topk{k}_rkl_olmo7bt"
        run = f"st_tt_clip005_topk{k}_rkl_c1024_olmo7bt"
        text = text.replace("st_tt_clip005_1e6_olmo7bt", job, 1)
        text = text.replace(
            "RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_openthoughts_olmo7bt}",
            f"RUN_NAME=${{RUN_NAME:-{run}}}",
        )
        text = text.replace(
            "export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-olmo3_7b_think_fullparam_100step_openthoughts}",
            f"export WANDB_RUN_GROUP=${{WANDB_RUN_GROUP:-olmo3_7b_think_topk{k}_rkl_openthoughts}}",
        )
        if "BETA=${BETA" not in text:
            text = text.replace(
                "LEARNING_RATE=${LEARNING_RATE:-1e-6}\n",
                "LEARNING_RATE=${LEARNING_RATE:-1e-6}\nBETA=${BETA:-1.0}\n",
                1,
            )
        text = text.replace(
            '  --learning-rate "${LEARNING_RATE}" \\\n  --jsd-token-clip',
            '  --learning-rate "${LEARNING_RATE}" \\\n  --beta "${BETA}" \\\n  --jsd-token-clip',
            1,
        )
        text = text.replace(
            '  --seed "${SEED}" \\\n  "${THINK_ARGS[@]}"',
            '  --top-k-loss ' + str(k) + ' \\\n  --seed "${SEED}" \\\n  "${THINK_ARGS[@]}"',
        )
        note = (
            f"# Reverse top-k KL (k={k}) for Olmo-3-7B-Think; rollout c1024; clip=0.05.\n"
            "# beta=1 → reverse: k=1 uses log π_S(x)−log π_T(x); k>1 uses student TopK + KL(p̃‖q̃).\n"
            "# (1.7b / 4b-it top-k stay forward: default beta=0, teacher TopK.)\n"
        )
        text = text.replace("set -euo pipefail\n\n", f"set -euo pipefail\n\n{note}", 1)
        text = text.replace(
            'echo "[launch] lr=${LEARNING_RATE} jsd_token_clip=${JSD_TOKEN_CLIP} seed=${SEED}"',
            'echo "[launch] lr=${LEARNING_RATE} beta=${BETA} jsd_token_clip=${JSD_TOKEN_CLIP} '
            f'seed=${{SEED}} top_k_loss={k} (reverse)"',
        )
        write(ROOT / f"scripts/train/olmo3_7b_think/topk/opsd_st_tt_clip005_topk{k}_c1024_ot.sh", text)

    # ----- topk 4b-instruct -----
    ref_4bi = (
        ROOT / "scripts/train/qwen3_4b_instruct/jsd005/opsd_student_nothink_teacher_nothink_clip005_1e_6_openthoughts.sh"
    ).read_text()
    if "SEED=${SEED" not in ref_4bi:
        ref_4bi = ref_4bi.replace(
            "JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}",
            "JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}\nSEED=${SEED:-42}",
            1,
        )
    if "--seed" not in ref_4bi:
        ref_4bi = ref_4bi.replace(
            '  "${THINK_ARGS[@]}"',
            '  --seed "${SEED}" \\\n  "${THINK_ARGS[@]}"',
            1,
        )
    for k in [1, 4, 16]:
        text = ref_4bi
        job = f"snt_tnt_clip005_topk{k}_oti"
        run = f"snt_tnt_clip005_topk{k}_c1024_oti"
        text = text.replace("snt_tnt_clip005_1e6_oti", job, 1)
        text = text.replace(
            "RUN_NAME=${RUN_NAME:-snt_tnt_clip005_1e_6_openthoughts_instruct}",
            f"RUN_NAME=${{RUN_NAME:-{run}}}",
        )
        text = text.replace(
            "export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_4b_instruct_fullparam_100step_openthoughts}",
            f"export WANDB_RUN_GROUP=${{WANDB_RUN_GROUP:-qwen3_4b_instruct_topk{k}_openthoughts}}",
        )
        text = text.replace(
            '  --seed "${SEED}" \\\n  "${THINK_ARGS[@]}"',
            '  --top-k-loss ' + str(k) + ' \\\n  --seed "${SEED}" \\\n  "${THINK_ARGS[@]}"',
        )
        note = (
            f"# Top-k KL (k={k}) for Qwen3-4B-Instruct (snt_tnt); rollout c1024; clip=0.05.\n"
            "# Naming uses snt_tnt (Instruct has no think mode).\n"
        )
        text = text.replace("set -euo pipefail\n\n", f"set -euo pipefail\n\n{note}", 1)
        write(ROOT / f"scripts/train/qwen3_4b_instruct/topk/opsd_snt_tnt_clip005_topk{k}_c1024_ot.sh", text)

    # ----- multi-seed train wrappers -----
    variants_1p7 = {
        "c256": "scripts/train/qwen3_1.7b/jsd005/length/opsd_st_tt_clip005_c256_1e6_ot.sh",
        "c1024": "scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts.sh",
        "answer": "scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_answer.sh",
        "ios": "scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_irrelevant_other_sol.sh",
        "first256": "scripts/train/qwen3_1.7b/jsd005/opsd_st_tt_clip005_c1024_first256_openthoughts.sh",
        "uni256": "scripts/train/qwen3_1.7b/jsd005/opsd_st_tt_clip005_c1024_uni256_openthoughts.sh",
        "last256": "scripts/train/qwen3_1.7b/jsd005/opsd_st_tt_clip005_c1024_last256_openthoughts.sh",
    }
    variants_olmo = {
        "c256": "scripts/train/olmo3_7b_think/jsd005/length/opsd_st_tt_clip005_c256_1e6_ot.sh",
        "c1024": "scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts.sh",
        "answer": "scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_answer.sh",
        "ios": "scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_irrelevant_other_sol.sh",
        "first256": "scripts/train/olmo3_7b_think/jsd005/opsd_st_tt_clip005_c1024_first256_openthoughts.sh",
        "uni256": "scripts/train/olmo3_7b_think/jsd005/opsd_st_tt_clip005_c1024_uni256_openthoughts.sh",
        "last256": "scripts/train/olmo3_7b_think/jsd005/opsd_st_tt_clip005_c1024_last256_openthoughts.sh",
    }

    for seed in [1024, 65536]:
        for name, parent in variants_1p7.items():
            run = f"st_tt_clip005_{name}_seed{seed}_1p7b"
            job = f"st_tt_{name}_s{seed}_1p7b"[:40]
            write(
                ROOT / f"scripts/train/qwen3_1.7b/jsd005/seed/opsd_st_tt_clip005_{name}_seed{seed}_ot.sh",
                make_seed_train_wrapper(
                    sbatch_header_1p7(job), parent, job, run, seed, f"# variant={name}"
                ),
            )
        for name, parent in variants_olmo.items():
            run = f"st_tt_clip005_{name}_seed{seed}_olmo7bt"
            job = f"st_tt_{name}_s{seed}_olmo"[:40]
            write(
                ROOT / f"scripts/train/olmo3_7b_think/jsd005/seed/opsd_st_tt_clip005_{name}_seed{seed}_ot.sh",
                make_seed_train_wrapper(
                    sbatch_header_olmo(job), parent, job, run, seed, f"# variant={name}"
                ),
            )

    # ----- multi-seed eval submit -----
    write(
        ROOT / "scripts/eval/1.7b/seed/submit_four_seed.sh",
        """#!/bin/bash
# Submit 4 think evals (aime24/25/26 + hmmt25) with a non-default SEED.
# Required: CHECKPOINT_PATH
# Optional: SEED (default 1024), EVAL_TAG, BASE_DIR
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASE_DIR=${BASE_DIR:-${ROOT}}
cd "${BASE_DIR}"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH}"
SEED=${SEED:-1024}
DATASETS=(aime24 aime25 aime26 hmmt25)

short_ds() {
  case "$1" in
    aime24) echo a24 ;;
    aime25) echo a25 ;;
    aime26) echo a26 ;;
    hmmt25) echo h25 ;;
    *) echo "$1" ;;
  esac
}

if [[ -z "${EVAL_TAG:-}" ]]; then
  _ckpt_base="$(basename "${CHECKPOINT_PATH}")"
  _job="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
  _run="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
  if [[ "${_ckpt_base}" == checkpoint-* ]]; then
    EVAL_TAG="${_run}_${_ckpt_base}_seed${SEED}"
  else
    EVAL_TAG="${_run}_${_job}_${_ckpt_base}_seed${SEED}"
  fi
fi

echo "[submit-seed-evals] 1.7b think ckpt=${CHECKPOINT_PATH} seed=${SEED} tag=${EVAL_TAG}"
SCRIPT_DIR="${BASE_DIR}/scripts/eval/1.7b"
LOG_ROOT="log/eval/1.7b"
for ds in "${DATASETS[@]}"; do
  sbatch \\
    --job-name="seed${SEED}_17_$(short_ds "${ds}")" \\
    --output="${LOG_ROOT}/${ds}/think/%x.%j.out" \\
    --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}",SEED="${SEED}",OUTPUT_JSON="${BASE_DIR}/eval_outputs/${EVAL_TAG}/${ds}_1.7b_think_seed${SEED}.json" \\
    "${SCRIPT_DIR}/${ds}_think.sh"
done
""",
    )
    for seed in [1024, 65536]:
        write(
            ROOT / f"scripts/eval/1.7b/seed/submit_four_seed{seed}.sh",
            f"""#!/bin/bash
# Submit 4 think evals with SEED={seed}. Requires CHECKPOINT_PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../../../.." && pwd)"
export SEED={seed}
exec bash "${{ROOT}}/scripts/eval/1.7b/seed/submit_four_seed.sh" "$@"
""",
        )

    write(
        ROOT / "scripts/eval/olmo_7b_think/seed/submit_four_seed.sh",
        """#!/bin/bash
# Submit 4 SGLang think evals with a non-default SEED for Olmo-3-7B-Think.
# Required: CHECKPOINT_PATH
# Optional: SEED (default 1024), EVAL_TAG, BASE_DIR
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASE_DIR=${BASE_DIR:-${ROOT}}
cd "${BASE_DIR}"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH}"
SEED=${SEED:-1024}
DATASETS=(aime24 aime25 aime26 hmmt25)

short_ds() {
  case "$1" in
    aime24) echo a24 ;;
    aime25) echo a25 ;;
    aime26) echo a26 ;;
    hmmt25) echo h25 ;;
    *) echo "$1" ;;
  esac
}

if [[ -z "${EVAL_TAG:-}" ]]; then
  _ckpt_base="$(basename "${CHECKPOINT_PATH}")"
  _job="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
  _run="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
  if [[ "${_ckpt_base}" == checkpoint-* ]]; then
    EVAL_TAG="${_run}_${_ckpt_base}_seed${SEED}"
  else
    EVAL_TAG="${_run}_${_job}_${_ckpt_base}_seed${SEED}"
  fi
fi

echo "[submit-seed-evals] olmo7bt ckpt=${CHECKPOINT_PATH} seed=${SEED} tag=${EVAL_TAG}"
SCRIPT_DIR="${BASE_DIR}/scripts/eval/olmo_7b_think"
LOG_ROOT="log/eval/olmo_7b_think"
for ds in "${DATASETS[@]}"; do
  sbatch \\
    --job-name="seed${SEED}_olmo_$(short_ds "${ds}")" \\
    --output="${LOG_ROOT}/${ds}/sgl/%x.%j.out" \\
    --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}",SEED="${SEED}",OUTPUT_JSON="${BASE_DIR}/eval_outputs/${EVAL_TAG}/${ds}_olmo_7b_think_sgl_seed${SEED}.json" \\
    "${SCRIPT_DIR}/${ds}_sgl.sh"
done
""",
    )
    for seed in [1024, 65536]:
        write(
            ROOT / f"scripts/eval/olmo_7b_think/seed/submit_four_seed{seed}.sh",
            f"""#!/bin/bash
# Submit 4 SGLang think evals with SEED={seed}. Requires CHECKPOINT_PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../../../.." && pwd)"
export SEED={seed}
exec bash "${{ROOT}}/scripts/eval/olmo_7b_think/seed/submit_four_seed.sh" "$@"
""",
        )

    write(
        ROOT / "scripts/eval/1.7b/topk/submit_four.sh",
        """#!/bin/bash
# Submit 4 think evals for a top-k KL 1.7b checkpoint (default SEED=42).
# Required: CHECKPOINT_PATH
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASE_DIR=${BASE_DIR:-${ROOT}}
: "${CHECKPOINT_PATH:?}"
SEED=${SEED:-42}
export SEED
exec bash "${BASE_DIR}/scripts/eval/1.7b/seed/submit_four_seed.sh"
""",
    )
    write(
        ROOT / "scripts/eval/olmo_7b_think/topk/submit_four.sh",
        """#!/bin/bash
# Submit 4 SGLang evals for a top-k KL olmo checkpoint (default SEED=42).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASE_DIR=${BASE_DIR:-${ROOT}}
: "${CHECKPOINT_PATH:?}"
SEED=${SEED:-42}
export SEED
exec bash "${BASE_DIR}/scripts/eval/olmo_7b_think/seed/submit_four_seed.sh"
""",
    )
    write(
        ROOT / "scripts/eval/4b_instruct/topk/submit_four.sh",
        """#!/bin/bash
# Submit 4 nothink evals for a top-k KL 4B-Instruct checkpoint (default SEED=42).
# Required: CHECKPOINT_PATH
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASE_DIR=${BASE_DIR:-${ROOT}}
cd "${BASE_DIR}"
: "${CHECKPOINT_PATH:?}"
SEED=${SEED:-42}
DATASETS=(aime24 aime25 aime26 hmmt25)

short_ds() {
  case "$1" in
    aime24) echo a24 ;; aime25) echo a25 ;; aime26) echo a26 ;; hmmt25) echo h25 ;; *) echo "$1" ;;
  esac
}

if [[ -z "${EVAL_TAG:-}" ]]; then
  _ckpt_base="$(basename "${CHECKPOINT_PATH}")"
  _job="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
  _run="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
  if [[ "${_ckpt_base}" == checkpoint-* ]]; then
    EVAL_TAG="${_run}_${_ckpt_base}"
  else
    EVAL_TAG="${_run}_${_job}_${_ckpt_base}"
  fi
fi

SCRIPT_DIR="${BASE_DIR}/scripts/eval/4b_instruct"
LOG_ROOT="log/eval/4b_instruct"
for ds in "${DATASETS[@]}"; do
  sbatch \\
    --job-name="topk_4bi_$(short_ds "${ds}")" \\
    --output="${LOG_ROOT}/${ds}/nothink/%x.%j.out" \\
    --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}",SEED="${SEED}",OUTPUT_JSON="${BASE_DIR}/eval_outputs/${EVAL_TAG}/${ds}_4b_instruct_nothink.json" \\
    "${SCRIPT_DIR}/${ds}_nothink.sh"
done
""",
    )

    for ds in ["aime24", "aime25", "aime26", "hmmt25"]:
        p = ROOT / f"scripts/eval/4b_instruct/{ds}_nothink.sh"
        text = p.read_text()
        if "SEED=${SEED" not in text:
            text = text.replace("set -euo pipefail\n", "set -euo pipefail\n\nSEED=${SEED:-42}\n", 1)
        if "  --seed 42 \\" in text:
            text = text.replace("  --seed 42 \\", '  --seed "${SEED}" \\', 1)
        p.write_text(text)
        print(f"[patched eval] {p.relative_to(ROOT)}")

    # Patch first256 1.7b for SEED if clone missed it (last256 already patched)
    p = ROOT / "scripts/train/qwen3_1.7b/jsd005/opsd_st_tt_clip005_c1024_first256_openthoughts.sh"
    text = p.read_text()
    if "SEED=${SEED" not in text:
        text = text.replace(
            "JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}",
            "JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}\nSEED=${SEED:-42}",
            1,
        )
    if "--seed" not in text:
        text = text.replace(
            '  "${THINK_ARGS[@]}"',
            '  --seed "${SEED}" \\\n  "${THINK_ARGS[@]}"',
            1,
        )
    p.write_text(text)

    print("\nDone.")


if __name__ == "__main__":
    main()
