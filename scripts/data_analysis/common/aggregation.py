"""Aggregate per-token metrics into summary statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np


class MetricsAggregator:
    """Accumulate divergence, preference, top-k hit, calibration, and clip stats."""

    def __init__(
        self,
        topk_names: tuple[str, ...] = ("k1", "k16"),
        topk_hit_ks: tuple[int, ...] = (4, 8, 16, 32, 64),
    ) -> None:
        self.topk_names = topk_names
        self.topk_hit_ks = topk_hit_ks
        self.max_hit_k = max(topk_hit_ks) if topk_hit_ks else 64

        self.n_tokens = 0
        self.n_rollouts = 0
        self.sum_forward_kl = 0.0
        self.sum_reverse_kl = 0.0
        self.sum_jsd_sym = 0.0
        self.sum_jsd = 0.0
        self.sum_jsd_clipped = 0.0
        self.n_jsd_clipped = 0
        self.sum_log_ratio = 0.0
        self.sum_advantage = 0.0
        self.sum_abs_advantage = 0.0
        self.sum_pos_advantage = 0.0
        self.sum_neg_advantage = 0.0
        self.n_encourage = 0
        self.n_discourage = 0
        self.n_tie = 0
        self.sum_snr = 0.0
        self.sum_entropy = 0.0
        self.sum_teacher_entropy = 0.0
        self.sum_entropy_gap = 0.0
        self.sum_s_logp = 0.0
        self.sum_t_logp = 0.0
        self.sum_s_conf_gap = 0.0
        self.sum_t_conf_gap = 0.0
        self.n_top1_agree = 0
        self.sum_topk: dict[str, float] = {k: 0.0 for k in topk_names}
        self.hit_s: dict[int, int] = {k: 0 for k in topk_hit_ks}
        self.hit_t: dict[int, int] = {k: 0 for k in topk_hit_ks}
        self.hit_t_sum_adv = 0.0
        self.miss_t_sum_adv = 0.0
        self.hit_t_n = 0
        self.miss_t_n = 0
        self.sum_rank_s_capped = 0.0
        self.sum_rank_t_capped = 0.0
        self.n_rank_s_known = 0
        self.n_rank_t_known = 0
        self.sum_rank_s_known = 0.0
        self.sum_rank_t_known = 0.0
        self.pos_hit_t = Counter()
        self.pos_n = Counter()
        self.pos_enc = Counter()
        self.pos_dec = Counter()
        self.rollout_frac_enc: list[float] = []

        self.token_jsd: Counter[int] = Counter()
        self.token_jsd_sum: dict[int, float] = defaultdict(float)
        self.disagree_student_pref: Counter[int] = Counter()
        self.disagree_teacher_pref: Counter[int] = Counter()
        self.disagree_pair: Counter[tuple[int, int]] = Counter()
        self.enc_count: Counter[int] = Counter()
        self.dec_count: Counter[int] = Counter()
        self.enc_adv_sum: dict[int, float] = defaultdict(float)
        self.dec_adv_sum: dict[int, float] = defaultdict(float)
        self.length_hist: Counter[str] = Counter()

    def update(self, metrics: dict[str, Any], *, window_label: str | None = None) -> None:
        n = metrics.get("n_tokens", 0)
        if n == 0:
            return
        self.n_rollouts += 1
        self.n_tokens += n

        jsd = metrics["jsd_kl"]
        jsd_clip = metrics.get("jsd_kl_clipped", jsd)
        would_clip = metrics.get("jsd_would_clip", [])
        forward_kl = metrics.get("forward_kl") or jsd
        reverse_kl = metrics.get("reverse_kl") or []
        jsd_sym = metrics.get("jsd_sym") or []
        lr = metrics["log_ratio_k1"]
        adv = metrics.get("advantage") or [-x for x in lr]
        snr = metrics.get("snr") or []
        s_ent = metrics.get("student_entropy") or []
        t_ent = metrics.get("teacher_entropy") or []
        ent_gap = metrics.get("entropy_gap") or []
        agree = metrics.get("top1_agree") or []
        s_arg = metrics.get("student_argmax_ids") or []
        t_arg = metrics.get("teacher_argmax_ids") or []
        token_ids = metrics.get("token_ids") or []

        self.sum_forward_kl += float(sum(forward_kl))
        if reverse_kl:
            self.sum_reverse_kl += float(sum(reverse_kl))
        if jsd_sym:
            self.sum_jsd_sym += float(sum(jsd_sym))
        self.sum_jsd += float(sum(jsd))
        self.sum_jsd_clipped += float(sum(jsd_clip))
        if would_clip:
            self.n_jsd_clipped += int(sum(1 for x in would_clip if x > 0))
        self.sum_log_ratio += float(sum(lr))
        self.sum_advantage += float(sum(adv))
        self.sum_abs_advantage += float(sum(abs(a) for a in adv))

        enc_mask = [a > 0 for a in adv]
        dec_mask = [a < 0 for a in adv]
        tie_mask = [not (e or d) for e, d in zip(enc_mask, dec_mask)]
        self.n_encourage += int(sum(enc_mask))
        self.n_discourage += int(sum(dec_mask))
        self.n_tie += int(sum(tie_mask))
        if any(enc_mask):
            self.sum_pos_advantage += float(sum(a for a, e in zip(adv, enc_mask) if e))
        if any(dec_mask):
            self.sum_neg_advantage += float(sum(a for a, d in zip(adv, dec_mask) if d))
        self.rollout_frac_enc.append(float(sum(enc_mask) / n))

        if snr:
            self.sum_snr += float(sum(snr))
        if s_ent:
            self.sum_entropy += float(sum(s_ent))
        if t_ent:
            self.sum_teacher_entropy += float(sum(t_ent))
        if ent_gap:
            self.sum_entropy_gap += float(sum(ent_gap))
        if metrics.get("student_logp"):
            self.sum_s_logp += float(sum(metrics["student_logp"]))
        if metrics.get("teacher_logp"):
            self.sum_t_logp += float(sum(metrics["teacher_logp"]))
        if metrics.get("student_confidence_gap"):
            self.sum_s_conf_gap += float(sum(metrics["student_confidence_gap"]))
        if metrics.get("teacher_confidence_gap"):
            self.sum_t_conf_gap += float(sum(metrics["teacher_confidence_gap"]))
        if agree:
            self.n_top1_agree += int(sum(1 for a in agree if a))

        for name in self.topk_names:
            vals = lr if name == "k1" else metrics.get("topk_kl", {}).get(name, [])
            self.sum_topk[name] += float(sum(vals))

        topk_hit = metrics.get("topk_hit", {})
        for k in self.topk_hit_ks:
            ks = str(k)
            if ks not in topk_hit:
                continue
            s_hits = topk_hit[ks].get("student", [])
            t_hits = topk_hit[ks].get("teacher", [])
            self.hit_s[k] += int(sum(1 for h in s_hits if h))
            self.hit_t[k] += int(sum(1 for h in t_hits if h))

        k_split = self.max_hit_k
        t_hit_k = topk_hit.get(str(k_split), {}).get("teacher", [])
        if t_hit_k:
            for i, (a, h) in enumerate(zip(adv, t_hit_k)):
                if h:
                    self.hit_t_sum_adv += float(a)
                    self.hit_t_n += 1
                else:
                    self.miss_t_sum_adv += float(a)
                    self.miss_t_n += 1
                bucket = "early" if i < n / 3 else ("mid" if i < 2 * n / 3 else "late")
                self.pos_n[bucket] += 1
                if h:
                    self.pos_hit_t[bucket] += 1

        ranks_s = metrics.get("rank_student_topk", [])
        ranks_t = metrics.get("rank_teacher_topk", [])
        for rs, rt in zip(ranks_s, ranks_t):
            rs_i, rt_i = int(rs), int(rt)
            if rs_i > 0:
                self.n_rank_s_known += 1
                self.sum_rank_s_known += rs_i
            self.sum_rank_s_capped += rs_i if rs_i > 0 else self.max_hit_k + 1
            if rt_i > 0:
                self.n_rank_t_known += 1
                self.sum_rank_t_known += rt_i
            self.sum_rank_t_capped += rt_i if rt_i > 0 else self.max_hit_k + 1

        for tid, j in zip(token_ids, jsd):
            tid = int(tid)
            self.token_jsd[tid] += 1
            self.token_jsd_sum[tid] += float(j)

        if s_arg and t_arg and len(s_arg) == len(t_arg):
            for sid, tid in zip(s_arg, t_arg):
                sid_i, tid_i = int(sid), int(tid)
                if sid_i != tid_i:
                    self.disagree_student_pref[sid_i] += 1
                    self.disagree_teacher_pref[tid_i] += 1
                    self.disagree_pair[(sid_i, tid_i)] += 1

        for tid, a, e, d in zip(token_ids, adv, enc_mask, dec_mask):
            tid_i = int(tid)
            if e:
                self.enc_count[tid_i] += 1
                self.enc_adv_sum[tid_i] += float(a)
            elif d:
                self.dec_count[tid_i] += 1
                self.dec_adv_sum[tid_i] += float(a)

        for i, a in enumerate(adv):
            bucket = "early" if i < n / 3 else ("mid" if i < 2 * n / 3 else "late")
            if a > 0:
                self.pos_enc[bucket] += 1
            elif a < 0:
                self.pos_dec[bucket] += 1

        if window_label:
            self.length_hist[window_label] += n

    def _top_adv_tokens(
        self,
        counts: Counter[int],
        adv_sum: dict[int, float],
        tokenizer: Any,
        k: int,
        *,
        reverse: bool,
    ) -> list[dict[str, Any]]:
        items = []
        for tid, cnt in counts.most_common(max(k * 5, k)):
            s = adv_sum.get(tid, 0.0)
            items.append(
                {
                    "token_id": int(tid),
                    "token": tokenizer.decode([tid]),
                    "count": int(cnt),
                    "sum_advantage": float(s),
                    "mean_advantage": float(s / cnt) if cnt else 0.0,
                }
            )
        items.sort(key=lambda x: x["sum_advantage"], reverse=reverse)
        return items[:k]

    def top_loss_tokens(self, tokenizer: Any, k: int = 50) -> list[dict[str, Any]]:
        items = []
        for tid, cnt in self.token_jsd.most_common(max(k, 1)):
            s = self.token_jsd_sum[tid]
            items.append(
                {
                    "token_id": tid,
                    "token": tokenizer.decode([tid]),
                    "count": int(cnt),
                    "sum_jsd_kl": float(s),
                    "mean_jsd_kl": float(s / cnt),
                }
            )
        return sorted(items, key=lambda x: x["sum_jsd_kl"], reverse=True)[:k]

    def _top_pref_tokens(self, counter: Counter[int], tokenizer: Any, k: int) -> list[dict[str, Any]]:
        return [
            {
                "token_id": int(tid),
                "token": tokenizer.decode([tid]),
                "count": int(cnt),
            }
            for tid, cnt in counter.most_common(k)
        ]

    def _top_disagree_pairs(self, tokenizer: Any, k: int) -> list[dict[str, Any]]:
        return [
            {
                "student_token_id": int(sid),
                "teacher_token_id": int(tid),
                "student_token": tokenizer.decode([sid]),
                "teacher_token": tokenizer.decode([tid]),
                "count": int(cnt),
            }
            for (sid, tid), cnt in self.disagree_pair.most_common(k)
        ]

    def summary(self, tokenizer: Any | None = None, top_token_k: int = 50) -> dict[str, Any]:
        n = max(self.n_tokens, 1)
        out: dict[str, Any] = {
            "n_rollouts": self.n_rollouts,
            "n_tokens": self.n_tokens,
            "mean_forward_kl": self.sum_forward_kl / n,
            "mean_reverse_kl": self.sum_reverse_kl / n,
            "mean_jsd_sym": self.sum_jsd_sym / n,
            "mean_jsd_kl": self.sum_jsd / n,
            "mean_jsd_kl_clipped": self.sum_jsd_clipped / n,
            "frac_jsd_clipped": self.n_jsd_clipped / n,
            "n_jsd_clipped": self.n_jsd_clipped,
            "mean_log_ratio_k1": self.sum_log_ratio / n,
            "mean_advantage": self.sum_advantage / n,
            "mean_abs_advantage": self.sum_abs_advantage / n,
            "mean_pos_advantage": (self.sum_pos_advantage / self.n_encourage) if self.n_encourage else 0.0,
            "mean_neg_advantage": (self.sum_neg_advantage / self.n_discourage) if self.n_discourage else 0.0,
            "n_encourage": self.n_encourage,
            "n_discourage": self.n_discourage,
            "n_tie": self.n_tie,
            "frac_encourage": self.n_encourage / n,
            "frac_discourage": self.n_discourage / n,
            "frac_tie": self.n_tie / n,
            "mean_rollout_frac_encourage": float(np.mean(self.rollout_frac_enc)) if self.rollout_frac_enc else 0.0,
            "position_encourage": dict(self.pos_enc),
            "position_discourage": dict(self.pos_dec),
            "mean_snr": self.sum_snr / n,
            "mean_student_entropy": self.sum_entropy / n,
            "mean_teacher_entropy": self.sum_teacher_entropy / n,
            "mean_entropy_gap": self.sum_entropy_gap / n,
            "mean_student_logp": self.sum_s_logp / n,
            "mean_teacher_logp": self.sum_t_logp / n,
            "mean_student_confidence_gap": self.sum_s_conf_gap / n,
            "mean_teacher_confidence_gap": self.sum_t_conf_gap / n,
            "top1_agree_rate": self.n_top1_agree / n,
            "n_top1_agree": self.n_top1_agree,
            "n_top1_disagree": self.n_tokens - self.n_top1_agree,
            "mean_topk_kl": {name: self.sum_topk[name] / n for name in self.topk_names},
            "topk_hit": {
                str(k): {
                    "frac_sampled_in_student_topk": self.hit_s[k] / n,
                    "frac_sampled_in_teacher_topk": self.hit_t[k] / n,
                    "n_hit_student": self.hit_s[k],
                    "n_hit_teacher": self.hit_t[k],
                }
                for k in self.topk_hit_ks
            },
            "teacher_hit_at_max_k": {
                "k": self.max_hit_k,
                "frac_hit": self.hit_t_n / n if self.max_hit_k else 0.0,
                "mean_advantage_when_hit": (self.hit_t_sum_adv / self.hit_t_n) if self.hit_t_n else 0.0,
                "mean_advantage_when_miss": (self.miss_t_sum_adv / self.miss_t_n) if self.miss_t_n else 0.0,
                "position_hit_frac": {
                    b: (self.pos_hit_t[b] / self.pos_n[b] if self.pos_n[b] else 0.0)
                    for b in ("early", "mid", "late")
                },
            },
            "rank_within_topk_max": {
                "max_k": self.max_hit_k,
                "frac_sampled_in_student_top_maxk": self.n_rank_s_known / n,
                "frac_sampled_in_teacher_top_maxk": self.n_rank_t_known / n,
                "mean_rank_student_when_in_top_maxk": (
                    self.sum_rank_s_known / self.n_rank_s_known if self.n_rank_s_known else None
                ),
                "mean_rank_teacher_when_in_top_maxk": (
                    self.sum_rank_t_known / self.n_rank_t_known if self.n_rank_t_known else None
                ),
                "mean_rank_student_capped": self.sum_rank_s_capped / n,
                "mean_rank_teacher_capped": self.sum_rank_t_capped / n,
            },
            "length_window_token_counts": dict(self.length_hist),
        }
        if tokenizer is not None:
            out["top_loss_dominant_tokens"] = self.top_loss_tokens(tokenizer, top_token_k)
            out["top_disagree_student_pref_tokens"] = self._top_pref_tokens(
                self.disagree_student_pref, tokenizer, top_token_k
            )
            out["top_disagree_teacher_pref_tokens"] = self._top_pref_tokens(
                self.disagree_teacher_pref, tokenizer, top_token_k
            )
            out["top_disagree_argmax_pairs"] = self._top_disagree_pairs(tokenizer, top_token_k)
            out["top_encourage_by_count"] = self._top_adv_tokens(
                self.enc_count, self.enc_adv_sum, tokenizer, top_token_k, reverse=True
            )
            out["top_discourage_by_count"] = self._top_adv_tokens(
                self.dec_count, self.dec_adv_sum, tokenizer, top_token_k, reverse=False
            )
            out["top_encourage_by_sum_adv"] = self._top_adv_tokens(
                self.enc_count, self.enc_adv_sum, tokenizer, top_token_k, reverse=True
            )
            out["top_discourage_by_sum_adv"] = sorted(
                self._top_adv_tokens(
                    self.dec_count, self.dec_adv_sum, tokenizer, top_token_k * 5, reverse=False
                ),
                key=lambda x: x["sum_advantage"],
            )[:top_token_k]
        return out


def merge_summaries(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"groups": summaries}
