#!/bin/bash
# Shared by lcb/evalplus auto-submit: trained judge job names match math/code generate logs.
# Original (qwen3-*/olmo*) keeps the judge script default job-name.
# Trained: EVAL_TAG=ocr_qwen3_4b_ckpt100 → ocr_qwen3_4b_lcb_j
code_eval_trained_judge_job_name() {
  local ds_short=$1
  local tag="${EVAL_TAG:-}"
  [[ -n "${tag}" ]] || return 0
  case "${tag}" in
    qwen3-*|olmo3-*|olmo-3-*)
      return 0
      ;;
  esac
  local prefix="${tag%_checkpoint-100}"
  prefix="${prefix%_ckpt100}"
  prefix="${prefix:0:44}"
  local name="${prefix}_${ds_short}_j"
  echo "${name:0:64}"
}
