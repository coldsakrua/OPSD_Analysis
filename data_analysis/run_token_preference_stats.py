#!/usr/bin/env python3
"""OPSD teacher→student token preference stats on student rollouts.

Setup (aligned with training collator / snt_tt opsd scripts)
----------------------------------------------------------
- Student & teacher weights: same Qwen3-4B
- Student chat template: enable_thinking=False (no-think)
- Teacher chat template: enable_thinking=True (think)
- Generation: temperature=1.1, top_p=0.95, top_k=20, max_new_tokens=1024
- Preference signal on each sampled completion token:
    advantage = log π_T(x) − log π_S(x)   (temperature=1.1, same as train)
    encourage (raise π_S):  advantage > 0
    discourage (lower π_S): advantage < 0

Three teacher prompt settings (identical problem rows across all):
  1) opsd_sol   : official OPSD teacher = problem + reference solution
                  + OFFICIAL_TRANSITION + boxed  (privilege_mode=opsd, field=solution)
  2) opsd_nogt  : official scaffold without reference solution
                  → NO_GT_TRANSITION only         (privilege_mode=opsd, field=none)
  3) same       : identical user text to student
                  = problem + boxed               (privilege_mode=same)

Pipeline
--------
1) Sample N prompts that fit max_prompt_length under the longest teacher template.
2) vLLM: N × n_rollouts student no-think completions (default 8192×2=16384).
3) HF: student logprobs once (shared), then teacher logprobs per setting.
4) Aggregate encourage / discourage token stats → summary + top lists.

Outputs under --output-dir:
  samples.jsonl
  rollouts.jsonl
  summary.json
  top_tokens_{setting}_{encourage|discourage}.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_collator import SelfDistillationDataCollator  # noqa: E402


SETTINGS = (
    {
        "name": "opsd_sol",
        "privilege_mode": "opsd",
        "teacher_privilege_field": "solution",
        "desc": "official OPSD: problem + reference solution + transition",
    },
    {
        "name": "opsd_nogt",
        "privilege_mode": "opsd",
        "teacher_privilege_field": "none",
        "desc": "official-style without reference solution (NO_GT transition)",
    },
    {
        "name": "same",
        "privilege_mode": "same",
        "teacher_privilege_field": "none",
        "desc": "same user prompt as student (problem + boxed only)",
    },
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model-path",
        default="/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b",
    )
    p.add_argument(
        "--dataset-path",
        default=str(
            ROOT
            / "data/openthoughts/preprocessed/openthoughts.opsd.solution.snothink_tthink.maxprompt1024.parquet"
        ),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-prompts", type=int, default=8192, help="Unique problems (shared across settings)")
    p.add_argument("--n-rollouts", type=int, default=2, help="Completions per prompt → total=num_prompts*n_rollouts")
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--max-completion-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.1, help="Gen + logprob temperature (train default)")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--gen-batch-hint", type=int, default=64)
    p.add_argument("--score-batch-size", type=int, default=2, help="HF score microbatch (prompt+completion)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--top-token-k", type=int, default=100, help="How many top tokens to dump per class")
    p.add_argument("--skip-generate", action="store_true", help="Reuse rollouts.jsonl in output-dir")
    p.add_argument("--skip-score", action="store_true", help="Only generate rollouts")
    p.add_argument(
        "--settings",
        nargs="+",
        default=[s["name"] for s in SETTINGS],
        choices=[s["name"] for s in SETTINGS],
    )
    return p.parse_args()


def make_collator(
    tokenizer: Any,
    *,
    privilege_mode: str,
    teacher_privilege_field: str,
    max_prompt_length: int,
) -> SelfDistillationDataCollator:
    return SelfDistillationDataCollator(
        tokenizer=tokenizer,
        max_length=max_prompt_length + 1024,
        max_prompt_length=max_prompt_length,
        privilege_mode=privilege_mode,
        teacher_privilege_field=teacher_privilege_field,
        student_thinking=False,
        teacher_thinking=True,
    )


def load_shared_samples(
    dataset_path: str,
    tokenizer: Any,
    num_prompts: int,
    max_prompt_length: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample problems that fit under EVERY teacher setting (binding = opsd_sol)."""
    df = pd.read_parquet(dataset_path, columns=["problem", "solution", "answer"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))

    collators = {
        s["name"]: make_collator(
            tokenizer,
            privilege_mode=s["privilege_mode"],
            teacher_privilege_field=s["teacher_privilege_field"],
            max_prompt_length=max_prompt_length,
        )
        for s in SETTINGS
    }

    out: list[dict[str, Any]] = []
    for idx in tqdm(order, desc="filter/sample prompts", total=len(order)):
        row = df.iloc[int(idx)]
        feature = {
            "problem": str(row["problem"]).strip(),
            "solution": str(row["solution"]).strip() if row["solution"] is not None else "",
            "answer": str(row["answer"]).strip() if row["answer"] is not None else "",
        }
        if not feature["problem"]:
            continue
        # Binding constraint: official+solution teacher think prompt must fit.
        if not collators["opsd_sol"].fits(feature):
            continue
        # Also require other settings (should be shorter, but keep strict).
        if not all(c.fits(feature) for c in collators.values()):
            continue

        # Student prompt is identical for all modes; teacher varies.
        student_prompt, _ = collators["opsd_sol"].format_prompts(feature)
        sps = [c.format_prompts(feature)[0] for c in collators.values()]
        if len(set(sps)) != 1:
            raise RuntimeError("student prompts diverged across settings; check collator")

        rec: dict[str, Any] = {
            "row_id": int(idx),
            "problem": feature["problem"],
            "solution": feature["solution"],
            "answer": feature["answer"],
            "student_prompt": student_prompt,
            "student_prompt_len": len(tokenizer(student_prompt, add_special_tokens=False)["input_ids"]),
        }
        for name, col in collators.items():
            _, tp = col.format_prompts(feature)
            rec[f"teacher_prompt_{name}"] = tp
            rec[f"teacher_prompt_len_{name}"] = len(
                tokenizer(tp, add_special_tokens=False)["input_ids"]
            )

        out.append(rec)
        if len(out) >= num_prompts:
            break

    if len(out) < num_prompts:
        raise RuntimeError(f"Only found {len(out)} prompts <= {max_prompt_length}; need {num_prompts}")
    return out


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_generation(
    model_path: str,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_completion_length + 64,
        disable_custom_all_reduce=True,
    )
    sampling = SamplingParams(
        n=args.n_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_completion_length,
        seed=args.seed,
    )

    prompts = [s["student_prompt"] for s in samples]
    print(
        f"[gen] prompts={len(prompts)} n_rollouts={args.n_rollouts} "
        f"→ total={len(prompts) * args.n_rollouts} max_tokens={args.max_completion_length}",
        flush=True,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)

    rollouts: list[dict[str, Any]] = []
    lengths: list[int] = []
    for s, o in zip(samples, outputs):
        for ri, cand in enumerate(o.outputs):
            tok_ids = list(cand.token_ids)
            lengths.append(len(tok_ids))
            rollouts.append(
                {
                    "row_id": s["row_id"],
                    "rollout_idx": ri,
                    "problem": s["problem"],
                    "solution": s["solution"],
                    "answer": s["answer"],
                    "student_prompt": s["student_prompt"],
                    "student_prompt_len": s["student_prompt_len"],
                    "teacher_prompt_opsd_sol": s["teacher_prompt_opsd_sol"],
                    "teacher_prompt_opsd_nogt": s["teacher_prompt_opsd_nogt"],
                    "teacher_prompt_same": s["teacher_prompt_same"],
                    "teacher_prompt_len_opsd_sol": s["teacher_prompt_len_opsd_sol"],
                    "teacher_prompt_len_opsd_nogt": s["teacher_prompt_len_opsd_nogt"],
                    "teacher_prompt_len_same": s["teacher_prompt_len_same"],
                    "completion_token_ids": tok_ids,
                    "completion_len": len(tok_ids),
                    "completion_text": cand.text,
                    "finish_reason": getattr(cand, "finish_reason", None),
                }
            )
    arr = np.asarray(lengths, dtype=np.float64)
    print(
        f"[gen] done in {time.time() - t0:.1f}s | n={len(rollouts)} "
        f"mean_len={arr.mean():.1f} median={np.median(arr):.1f} "
        f"frac_hit_max={(arr >= args.max_completion_length - 1).mean():.3f}",
        flush=True,
    )
    del llm
    torch.cuda.empty_cache()
    return rollouts


@torch.no_grad()
def completion_logprobs_batch(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prompt_texts: list[str],
    completion_ids_list: list[list[int]],
    temperature: float,
) -> list[torch.Tensor]:
    """Right-pad batch; return per-example CPU float32 logprob vectors."""
    assert len(prompt_texts) == len(completion_ids_list)
    if not prompt_texts:
        return []

    prompt_ids_list = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompt_texts]
    seqs = [p + c for p, c in zip(prompt_ids_list, completion_ids_list)]
    plens = [len(p) for p in prompt_ids_list]
    clens = [len(c) for c in completion_ids_list]
    max_len = max(len(s) for s in seqs)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    batch = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=model.device)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=model.device)
    for i, s in enumerate(seqs):
        batch[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=model.device)
        attn[i, : len(s)] = 1

    out = model(input_ids=batch, attention_mask=attn)
    logits = out.logits.float() / temperature  # [B, L, V]

    results: list[torch.Tensor] = []
    for i, (plen, clen) in enumerate(zip(plens, clens)):
        if clen == 0:
            results.append(torch.empty(0, dtype=torch.float32))
            continue
        # positions predicting completion tokens
        pos_logits = logits[i, plen - 1 : plen - 1 + clen, :]
        logp = F.log_softmax(pos_logits, dim=-1)
        tok = batch[i, plen : plen + clen]
        results.append(logp.gather(-1, tok.unsqueeze(-1)).squeeze(-1).cpu())
    return results


class TokenAgg:
    """Accumulate encourage / discourage token stats."""

    def __init__(self) -> None:
        self.n_tokens = 0
        self.n_encourage = 0
        self.n_discourage = 0
        self.n_tie = 0
        self.sum_adv = 0.0
        self.sum_abs_adv = 0.0
        self.sum_pos_adv = 0.0
        self.sum_neg_adv = 0.0
        self.enc_count: Counter[int] = Counter()
        self.dec_count: Counter[int] = Counter()
        self.enc_adv_sum: dict[int, float] = defaultdict(float)
        self.dec_adv_sum: dict[int, float] = defaultdict(float)
        # position buckets (completion index)
        self.pos_enc = Counter()
        self.pos_dec = Counter()
        self.n_rollouts = 0
        self.rollout_frac_enc: list[float] = []

    def update(self, token_ids: list[int], advantage: torch.Tensor) -> None:
        if advantage.numel() == 0:
            return
        self.n_rollouts += 1
        adv = advantage.numpy()
        n = len(adv)
        self.n_tokens += n
        self.sum_adv += float(adv.sum())
        self.sum_abs_adv += float(np.abs(adv).sum())

        enc = adv > 0
        dec = adv < 0
        tie = ~(enc | dec)
        self.n_encourage += int(enc.sum())
        self.n_discourage += int(dec.sum())
        self.n_tie += int(tie.sum())
        self.sum_pos_adv += float(adv[enc].sum()) if enc.any() else 0.0
        self.sum_neg_adv += float(adv[dec].sum()) if dec.any() else 0.0
        self.rollout_frac_enc.append(float(enc.mean()) if n else 0.0)

        for t, a, e, d in zip(token_ids, adv, enc, dec):
            tid = int(t)
            if e:
                self.enc_count[tid] += 1
                self.enc_adv_sum[tid] += float(a)
            elif d:
                self.dec_count[tid] += 1
                self.dec_adv_sum[tid] += float(a)

        # coarse position: early / mid / late thirds
        for i, a in enumerate(adv):
            bucket = "early" if i < n / 3 else ("mid" if i < 2 * n / 3 else "late")
            if a > 0:
                self.pos_enc[bucket] += 1
            elif a < 0:
                self.pos_dec[bucket] += 1

    def summary(self, tokenizer: Any, top_k: int) -> dict[str, Any]:
        def top_list(counts: Counter[int], adv_sum: dict[int, float], k: int) -> list[dict]:
            items = []
            for tid, cnt in counts.most_common(k):
                s = adv_sum.get(tid, 0.0)
                items.append(
                    {
                        "token_id": tid,
                        "token": tokenizer.decode([tid]),
                        "token_repr": repr(tokenizer.decode([tid])),
                        "count": int(cnt),
                        "sum_advantage": float(s),
                        "mean_advantage": float(s / cnt) if cnt else 0.0,
                    }
                )
            return items

        n = max(self.n_tokens, 1)
        return {
            "n_rollouts": self.n_rollouts,
            "n_tokens": self.n_tokens,
            "n_encourage": self.n_encourage,
            "n_discourage": self.n_discourage,
            "n_tie": self.n_tie,
            "frac_encourage": self.n_encourage / n,
            "frac_discourage": self.n_discourage / n,
            "frac_tie": self.n_tie / n,
            "mean_advantage": self.sum_adv / n,
            "mean_abs_advantage": self.sum_abs_adv / n,
            "mean_pos_advantage": (self.sum_pos_adv / self.n_encourage) if self.n_encourage else 0.0,
            "mean_neg_advantage": (self.sum_neg_adv / self.n_discourage) if self.n_discourage else 0.0,
            "mean_rollout_frac_encourage": float(np.mean(self.rollout_frac_enc)) if self.rollout_frac_enc else 0.0,
            "position_encourage": dict(self.pos_enc),
            "position_discourage": dict(self.pos_dec),
            "top_encourage_by_count": top_list(self.enc_count, self.enc_adv_sum, top_k),
            "top_discourage_by_count": top_list(self.dec_count, self.dec_adv_sum, top_k),
            "top_encourage_by_sum_adv": sorted(
                top_list(self.enc_count, self.enc_adv_sum, max(top_k * 5, top_k)),
                key=lambda x: x["sum_advantage"],
                reverse=True,
            )[:top_k],
            "top_discourage_by_sum_adv": sorted(
                top_list(self.dec_count, self.dec_adv_sum, max(top_k * 5, top_k)),
                key=lambda x: x["sum_advantage"],  # most negative first
            )[:top_k],
        }


def score_setting(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    rollouts: list[dict[str, Any]],
    setting_name: str,
    student_logprobs_cache: list[torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    teacher_key = f"teacher_prompt_{setting_name}"
    agg = TokenAgg()
    bs = max(1, int(args.score_batch_size))

    for start in tqdm(range(0, len(rollouts), bs), desc=f"score/{setting_name}"):
        batch = rollouts[start : start + bs]
        # student logprobs (cached)
        s_logps = student_logprobs_cache[start : start + bs]

        t_prompts = [r[teacher_key] for r in batch]
        comps = [r["completion_token_ids"] for r in batch]
        t_logps = completion_logprobs_batch(
            model, tokenizer, t_prompts, comps, temperature=args.temperature
        )

        for r, s_lp, t_lp in zip(batch, s_logps, t_logps):
            L = min(len(r["completion_token_ids"]), s_lp.numel(), t_lp.numel())
            if L == 0:
                continue
            adv = t_lp[:L] - s_lp[:L]
            agg.update(r["completion_token_ids"][:L], adv)

    return agg.summary(tokenizer, args.top_token_k)


def compute_student_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    rollouts: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[torch.Tensor]:
    cache: list[torch.Tensor] = []
    bs = max(1, int(args.score_batch_size))
    for start in tqdm(range(0, len(rollouts), bs), desc="score/student"):
        batch = rollouts[start : start + bs]
        prompts = [r["student_prompt"] for r in batch]
        comps = [r["completion_token_ids"] for r in batch]
        cache.extend(
            completion_logprobs_batch(model, tokenizer, prompts, comps, temperature=args.temperature)
        )
    return cache


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] model={args.model_path}", flush=True)
    print(f"[cfg] dataset={args.dataset_path}", flush=True)
    print(
        f"[cfg] num_prompts={args.num_prompts} n_rollouts={args.n_rollouts} "
        f"total={args.num_prompts * args.n_rollouts} max_completion={args.max_completion_length}",
        flush=True,
    )
    print(f"[cfg] settings={args.settings} temperature={args.temperature}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples_path = out_dir / "samples.jsonl"
    rollouts_path = out_dir / "rollouts.jsonl"

    if args.skip_generate and rollouts_path.is_file():
        print(f"[gen] reuse {rollouts_path}", flush=True)
        rollouts = load_jsonl(rollouts_path)
        # rebuild samples meta from rollouts if needed
        if not samples_path.is_file():
            seen = set()
            samples = []
            for r in rollouts:
                if r["row_id"] in seen:
                    continue
                seen.add(r["row_id"])
                samples.append({k: r[k] for k in r if not k.startswith("completion") and k != "rollout_idx" and k != "finish_reason"})
            save_jsonl(samples_path, samples)
    else:
        samples = load_shared_samples(
            args.dataset_path,
            tokenizer,
            args.num_prompts,
            args.max_prompt_length,
            args.seed,
        )
        save_jsonl(samples_path, samples)
        print(f"[sample] wrote {len(samples)} prompts → {samples_path}", flush=True)

        # dump one example prompt per setting for audit
        ex = samples[0]
        audit = {
            "row_id": ex["row_id"],
            "student_prompt_tail": ex["student_prompt"][-400:],
            "teacher_prompt_opsd_sol_tail": ex["teacher_prompt_opsd_sol"][-400:],
            "teacher_prompt_opsd_nogt_tail": ex["teacher_prompt_opsd_nogt"][-400:],
            "teacher_prompt_same_tail": ex["teacher_prompt_same"][-400:],
            "lens": {
                "student": ex["student_prompt_len"],
                "opsd_sol": ex["teacher_prompt_len_opsd_sol"],
                "opsd_nogt": ex["teacher_prompt_len_opsd_nogt"],
                "same": ex["teacher_prompt_len_same"],
            },
        }
        (out_dir / "prompt_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

        rollouts = run_generation(args.model_path, samples, args)
        save_jsonl(rollouts_path, rollouts)
        print(f"[gen] wrote {len(rollouts)} rollouts → {rollouts_path}", flush=True)

    expected = args.num_prompts * args.n_rollouts
    if len(rollouts) != expected and not args.skip_generate:
        print(f"[warn] expected {expected} rollouts, got {len(rollouts)}", flush=True)

    if args.skip_score:
        print("[score] skipped", flush=True)
        return

    print("[score] loading HF model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()

    print("[score] student logprobs (shared across settings)...", flush=True)
    student_cache = compute_student_logprobs(model, tokenizer, rollouts, args)

    summary: dict[str, Any] = {
        "config": {
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "num_prompts": args.num_prompts,
            "n_rollouts": args.n_rollouts,
            "n_rollouts_actual": len(rollouts),
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "student_thinking": False,
            "teacher_thinking": True,
            "advantage": "logp_teacher(token) - logp_student(token)",
            "encourage": "advantage > 0  (teacher wants higher π_student)",
            "discourage": "advantage < 0 (teacher wants lower π_student)",
            "settings": {s["name"]: s["desc"] for s in SETTINGS if s["name"] in args.settings},
        },
        "settings": {},
    }

    for s in SETTINGS:
        if s["name"] not in args.settings:
            continue
        print(f"[score] setting={s['name']}: {s['desc']}", flush=True)
        stats = score_setting(model, tokenizer, rollouts, s["name"], student_cache, args)
        summary["settings"][s["name"]] = {"desc": s["desc"], **stats}
        # also dump top lists separately for easy viewing
        for kind in ("encourage", "discourage"):
            key = f"top_{kind}_by_count"
            path = out_dir / f"top_tokens_{s['name']}_{kind}.json"
            path.write_text(json.dumps(stats[key], ensure_ascii=False, indent=2), encoding="utf-8")

        # concise stdout
        print(
            f"  [{s['name']}] tokens={stats['n_tokens']} "
            f"encourage={stats['frac_encourage']:.3f} discourage={stats['frac_discourage']:.3f} "
            f"mean_adv={stats['mean_advantage']:.4f}",
            flush=True,
        )
        print(f"  top encourage: {[x['token_repr'] for x in stats['top_encourage_by_count'][:10]]}", flush=True)
        print(f"  top discourage: {[x['token_repr'] for x in stats['top_discourage_by_count'][:10]]}", flush=True)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] summary → {summary_path}", flush=True)


if __name__ == "__main__":
    main()
