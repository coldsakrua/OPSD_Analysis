#!/usr/bin/env bash
# Watch Q35/4B LoRA evals until complete (n=30, partial=false), then rebuild summary.
set -euo pipefail
BASE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE"
TARGET_RUNS=(
  snt_tt_lora_clip005_lr5e6_q35_4b_ckpt100
  st_tt_lora_clip005_lr5e6_q35_4b_ckpt100
  snt_tnt_lora_clip005_lr5e6_4b_ckpt100
)
STATUS_JSON=eval_outputs/_unified_summary/lora_watch_status.json

check_complete() {
  python3 - <<'PY'
import json, glob, os, sys
runs = [
  "snt_tt_lora_clip005_lr5e6_q35_4b_ckpt100",
  "st_tt_lora_clip005_lr5e6_q35_4b_ckpt100",
  "snt_tnt_lora_clip005_lr5e6_4b_ckpt100",
]
# Expected files
expect = {
  "snt_tt_lora_clip005_lr5e6_q35_4b_ckpt100": [
    "aime24_qwen35_4b_nothink","aime25_qwen35_4b_nothink","aime26_qwen35_4b_nothink","hmmt25_qwen35_4b_nothink"
  ],
  "st_tt_lora_clip005_lr5e6_q35_4b_ckpt100": [
    "aime24_qwen35_4b_think","aime25_qwen35_4b_think","aime26_qwen35_4b_think","hmmt25_qwen35_4b_think"
  ],
  "snt_tnt_lora_clip005_lr5e6_4b_ckpt100": [
    "aime24_4b_nothink","aime25_4b_nothink","aime26_4b_nothink","hmmt25_4b_nothink"
  ],
}
status = {}
all_ok = True
for run, stems in expect.items():
    items = []
    ok = True
    for stem in stems:
        path = f"eval_outputs/{run}/{stem}.metrics.json"
        if not os.path.exists(path):
            items.append({"file": stem, "n": 0, "partial": True, "ok": False})
            ok = False
            continue
        m = json.load(open(path))
        n = m.get("num_problems")
        partial = bool(m.get("partial_only"))
        good = (n == 30 and not partial)
        items.append({
            "file": stem, "n": n, "partial": partial,
            "pass1": m.get("avg1_pct"), "avg": m.get("average_correct_pct"), "ok": good,
        })
        if not good:
            ok = False
    status[run] = {"complete": ok, "items": items}
    all_ok = all_ok and ok
out = {"all_complete": all_ok, "runs": status}
print(json.dumps(out))
sys.exit(0 if all_ok else 1)
PY
}

echo "[watch] start $(date -Is)"
while true; do
  if OUT=$(check_complete); then
    echo "$OUT" > "$STATUS_JSON"
    echo "[watch] all target LoRA evals complete $(date -Is)"
    python3 scripts/analysis/rebuild_opsd_unified_summary.py
    # emit canvas patch numbers
    python3 - <<'PY'
import json
from pathlib import Path
canvas = json.load(open("eval_outputs/_unified_summary/canvas_key_runs.json"))
want = {
  "snt_tt_lora_clip005_lr5e6_q35_4b_ckpt100",
  "st_tt_lora_clip005_lr5e6_q35_4b_ckpt100",
  "snt_tnt_lora_clip005_lr5e6_4b_ckpt100",
}
for r in canvas:
    if r["run"] in want:
        print(r["run"], "nt_mean", r.get("nt_mean_pass1"), "delta", r.get("delta_nt_mean"),
              "a24", r.get("aime24_nt_pass1"), "a25", r.get("aime25_nt_pass1"),
              "a26", r.get("aime26_nt_pass1"), "h25", r.get("hmmt25_nt_pass1"),
              "th_a24", r.get("aime24_th_pass1"))
Path("eval_outputs/_unified_summary/lora_update_ready.flag").write_text("ready\n")
PY
    exit 0
  else
    echo "$OUT" > "$STATUS_JSON"
    echo "[watch] still partial $(date -Is)"
    # show compact progress
    python3 -c "import json; d=json.load(open('$STATUS_JSON'));
[print(k, [(i['file'].split('_')[0], i['n']) for i in v['items']]) for k,v in d['runs'].items()]"
  fi
  sleep 180
done
