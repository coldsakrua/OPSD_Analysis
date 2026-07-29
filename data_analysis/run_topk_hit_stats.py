#!/usr/bin/env python3
"""Top-k hit stats for the *sampled student token* on student rollouts.

Community reference (verl OPD / arxiv:2604.13016 / siyan-zhao OPSD top_k_loss):
  - Apply temperature to logits, then TopK (monotonic ⇒ topk(logits)==topk(log_softmax)).
  - Diagnostics focus on student-visited states (on-policy positions).

This script only scores the current sampled student token x_t (NOT full top-k set overlap):
  for K in {4,8,16,32,64}:
    - frac(x_t ∈ TopK_student(K))
    - frac(x_t ∈ TopK_teacher(K))
  plus mean log π_S(x_t), log π_T(x_t), advantage, and hit/miss splits.

Reuses rollouts.jsonl from a prior token_pref run (same prompts / completions).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data_analysis"))

# Reuse helpers / setting names from the preference script.
from run_token_preference_stats import (  # noqa: E402
    SETTINGS,
    load_jsonl,
)


DEFAULT_KS = (4, 8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model-path",
        default="/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b",
    )
    p.add_argument(
        "--rollouts-path",
        default=str(ROOT / "data_analysis/outputs/token_pref_2963814/rollouts.jsonl"),
        help="Existing student rollouts (must include student/teacher prompts + completion_token_ids)",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--score-batch-size", type=int, default=2)
    p.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument(
        "--settings",
        nargs="+",
        default=[s["name"] for s in SETTINGS],
        choices=[s["name"] for s in SETTINGS],
    )
    p.add_argument("--max-rollouts", type=int, default=0, help="0 = all")
    return p.parse_args()


class TopKHitAgg:
    """Accumulate sampled-token top-k hit rates for several K."""

    def __init__(self, ks: list[int]) -> None:
        self.ks = sorted(ks)
        self.max_k = max(self.ks)
        self.n_tokens = 0
        self.n_rollouts = 0
        self.sum_s_logp = 0.0
        self.sum_t_logp = 0.0
        self.sum_adv = 0.0
        self.hit_s = {k: 0 for k in self.ks}
        self.hit_t = {k: 0 for k in self.ks}
        # Conditional advantage / logp on teacher hit vs miss (largest K for split)
        self.k_split = self.max_k
        self.hit_t_sum_adv = 0.0
        self.miss_t_sum_adv = 0.0
        self.hit_t_n = 0
        self.miss_t_n = 0
        # Rank within top-max_k if present (1-indexed); else max_k+1 sentinel for mean_rank_capped
        self.sum_rank_s = 0.0
        self.sum_rank_t = 0.0
        self.n_rank_s_known = 0
        self.n_rank_t_known = 0
        # Early / mid / late teacher hit@max_k
        self.pos_hit_t = {"early": 0, "mid": 0, "late": 0}
        self.pos_n = {"early": 0, "mid": 0, "late": 0}

    def update(
        self,
        *,
        s_logp: torch.Tensor,
        t_logp: torch.Tensor,
        in_s: dict[int, torch.Tensor],
        in_t: dict[int, torch.Tensor],
        rank_s: torch.Tensor,
        rank_t: torch.Tensor,
    ) -> None:
        """All tensors length L (completion). ranks: 1..max_k or 0 if outside top-max_k."""
        L = int(s_logp.numel())
        if L == 0:
            return
        self.n_rollouts += 1
        self.n_tokens += L
        adv = (t_logp - s_logp).numpy()
        self.sum_s_logp += float(s_logp.sum())
        self.sum_t_logp += float(t_logp.sum())
        self.sum_adv += float(adv.sum())

        for k in self.ks:
            self.hit_s[k] += int(in_s[k].sum().item())
            self.hit_t[k] += int(in_t[k].sum().item())

        hit_mask = in_t[self.k_split].numpy().astype(bool)
        self.hit_t_n += int(hit_mask.sum())
        self.miss_t_n += int((~hit_mask).sum())
        if hit_mask.any():
            self.hit_t_sum_adv += float(adv[hit_mask].sum())
        if (~hit_mask).any():
            self.miss_t_sum_adv += float(adv[~hit_mask].sum())

        rs = rank_s.numpy()
        rt = rank_t.numpy()
        known_s = rs > 0
        known_t = rt > 0
        self.n_rank_s_known += int(known_s.sum())
        self.n_rank_t_known += int(known_t.sum())
        if known_s.any():
            self.sum_rank_s += float(rs[known_s].sum())
        if known_t.any():
            self.sum_rank_t += float(rt[known_t].sum())
        # capped mean rank: unknown → max_k+1
        capped_s = np.where(known_s, rs, self.max_k + 1)
        capped_t = np.where(known_t, rt, self.max_k + 1)
        # store via running sum into dedicated fields
        if not hasattr(self, "sum_rank_s_capped"):
            self.sum_rank_s_capped = 0.0
            self.sum_rank_t_capped = 0.0
        self.sum_rank_s_capped += float(capped_s.sum())
        self.sum_rank_t_capped += float(capped_t.sum())

        for i in range(L):
            bucket = "early" if i < L / 3 else ("mid" if i < 2 * L / 3 else "late")
            self.pos_n[bucket] += 1
            if hit_mask[i]:
                self.pos_hit_t[bucket] += 1

    def summary(self) -> dict[str, Any]:
        n = max(self.n_tokens, 1)
        per_k = {}
        for k in self.ks:
            per_k[str(k)] = {
                "frac_sampled_in_student_topk": self.hit_s[k] / n,
                "frac_sampled_in_teacher_topk": self.hit_t[k] / n,
                "n_hit_student": self.hit_s[k],
                "n_hit_teacher": self.hit_t[k],
            }
        return {
            "n_rollouts": self.n_rollouts,
            "n_tokens": self.n_tokens,
            "mean_student_logp": self.sum_s_logp / n,
            "mean_teacher_logp": self.sum_t_logp / n,
            "mean_advantage": self.sum_adv / n,
            "per_k": per_k,
            "teacher_hit_at_max_k": {
                "k": self.k_split,
                "frac_hit": self.hit_t_n / n,
                "mean_adv_when_hit": (self.hit_t_sum_adv / self.hit_t_n) if self.hit_t_n else 0.0,
                "mean_adv_when_miss": (self.miss_t_sum_adv / self.miss_t_n) if self.miss_t_n else 0.0,
                "position_hit_frac": {
                    b: (self.pos_hit_t[b] / self.pos_n[b] if self.pos_n[b] else 0.0)
                    for b in ("early", "mid", "late")
                },
            },
            "rank_within_topk_max": {
                "max_k": self.max_k,
                "frac_sampled_in_student_top_maxk": self.n_rank_s_known / n,
                "frac_sampled_in_teacher_top_maxk": self.n_rank_t_known / n,
                "mean_rank_student_when_in_top_maxk": (
                    self.sum_rank_s / self.n_rank_s_known if self.n_rank_s_known else None
                ),
                "mean_rank_teacher_when_in_top_maxk": (
                    self.sum_rank_t / self.n_rank_t_known if self.n_rank_t_known else None
                ),
                "mean_rank_student_capped": getattr(self, "sum_rank_s_capped", 0.0) / n,
                "mean_rank_teacher_capped": getattr(self, "sum_rank_t_capped", 0.0) / n,
                "note": "capped rank uses max_k+1 when token is outside TopK(max_k)",
            },
        }


@torch.no_grad()
def forward_sampled_topk(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prompt_texts: list[str],
    completion_ids_list: list[list[int]],
    temperature: float,
    ks: list[int],
) -> list[dict[str, Any]]:
    """Return per-example dicts with logp + topk hit masks for sampled tokens."""
    assert len(prompt_texts) == len(completion_ids_list)
    if not prompt_texts:
        return []

    max_k = max(ks)
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
    # Temperature before topk — matches OPSD / verl distillation path.
    logits = out.logits.float() / temperature

    results: list[dict[str, Any]] = []
    for i, (plen, clen) in enumerate(zip(plens, clens)):
        if clen == 0:
            results.append(
                {
                    "logp": torch.empty(0),
                    "in_topk": {k: torch.empty(0, dtype=torch.bool) for k in ks},
                    "rank": torch.empty(0, dtype=torch.long),
                }
            )
            continue
        pos_logits = logits[i, plen - 1 : plen - 1 + clen, :]  # [L, V]
        tok = batch[i, plen : plen + clen]  # [L]
        logp = F.log_softmax(pos_logits, dim=-1).gather(-1, tok.unsqueeze(-1)).squeeze(-1).cpu()

        topk_idx = torch.topk(pos_logits, k=max_k, dim=-1).indices  # [L, max_k]
        # rank: position of tok in topk (1..max_k), else 0
        eq = topk_idx == tok.unsqueeze(-1)  # [L, max_k]
        in_max = eq.any(dim=-1)
        # argmax over bool → first True index; garbage if none
        first = eq.float().argmax(dim=-1)
        rank = torch.where(in_max, first + 1, torch.zeros_like(first)).cpu()

        in_topk = {}
        for k in ks:
            in_topk[k] = (topk_idx[:, :k] == tok.unsqueeze(-1)).any(dim=-1).cpu()

        results.append({"logp": logp, "in_topk": in_topk, "rank": rank})
    return results


def score_setting(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    rollouts: list[dict[str, Any]],
    setting_name: str,
    student_cache: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    teacher_key = f"teacher_prompt_{setting_name}"
    agg = TopKHitAgg(args.ks)
    bs = max(1, int(args.score_batch_size))

    for start in tqdm(range(0, len(rollouts), bs), desc=f"topk/{setting_name}"):
        batch = rollouts[start : start + bs]
        s_pack = student_cache[start : start + bs]
        t_prompts = [r[teacher_key] for r in batch]
        comps = [r["completion_token_ids"] for r in batch]
        t_pack = forward_sampled_topk(
            model, tokenizer, t_prompts, comps, args.temperature, args.ks
        )

        for s, t in zip(s_pack, t_pack):
            L = min(s["logp"].numel(), t["logp"].numel())
            if L == 0:
                continue
            agg.update(
                s_logp=s["logp"][:L],
                t_logp=t["logp"][:L],
                in_s={k: s["in_topk"][k][:L] for k in args.ks},
                in_t={k: t["in_topk"][k][:L] for k in args.ks},
                rank_s=s["rank"][:L],
                rank_t=t["rank"][:L],
            )
    return agg.summary()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollouts_path = Path(args.rollouts_path)
    if not rollouts_path.is_file():
        raise FileNotFoundError(f"missing rollouts: {rollouts_path}")

    print(f"[cfg] model={args.model_path}", flush=True)
    print(f"[cfg] rollouts={rollouts_path}", flush=True)
    print(f"[cfg] ks={args.ks} settings={args.settings} T={args.temperature}", flush=True)

    rollouts = load_jsonl(rollouts_path)
    if args.max_rollouts and args.max_rollouts > 0:
        rollouts = rollouts[: args.max_rollouts]
    print(f"[data] n_rollouts={len(rollouts)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[score] loading HF model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()

    print("[score] student top-k / logp (shared)...", flush=True)
    t0 = time.time()
    student_cache: list[dict[str, Any]] = []
    bs = max(1, int(args.score_batch_size))
    for start in tqdm(range(0, len(rollouts), bs), desc="topk/student"):
        batch = rollouts[start : start + bs]
        student_cache.extend(
            forward_sampled_topk(
                model,
                tokenizer,
                [r["student_prompt"] for r in batch],
                [r["completion_token_ids"] for r in batch],
                args.temperature,
                args.ks,
            )
        )
    print(f"[score] student done in {time.time() - t0:.1f}s", flush=True)

    summary: dict[str, Any] = {
        "config": {
            "model_path": args.model_path,
            "rollouts_path": str(rollouts_path),
            "n_rollouts": len(rollouts),
            "temperature": args.temperature,
            "ks": args.ks,
            "metric": (
                "For each on-policy sampled token x_t: "
                "whether x_t ∈ TopK_π(K) for student and teacher. "
                "Matches verl/OPSD: temperature then topk on logits."
            ),
            "not_computed": (
                "Full TopK_T ∩ TopK_S overlap_ratio (needs entire top-k sets; "
                "user requested sampled-token-only stats)."
            ),
        },
        "settings": {},
    }

    for s in SETTINGS:
        if s["name"] not in args.settings:
            continue
        print(f"[score] setting={s['name']}", flush=True)
        stats = score_setting(model, tokenizer, rollouts, s["name"], student_cache, args)
        summary["settings"][s["name"]] = {"desc": s["desc"], **stats}
        # concise print
        row = " ".join(
            f"K={k}:S={stats['per_k'][str(k)]['frac_sampled_in_student_topk']:.3f}/"
            f"T={stats['per_k'][str(k)]['frac_sampled_in_teacher_topk']:.3f}"
            for k in args.ks
        )
        print(f"  [{s['name']}] {row}", flush=True)
        print(
            f"  mean_logp S={stats['mean_student_logp']:.4f} T={stats['mean_teacher_logp']:.4f} "
            f"A={stats['mean_advantage']:.4f}",
            flush=True,
        )

    out_path = out_dir / "topk_hit_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] → {out_path}", flush=True)


if __name__ == "__main__":
    main()
