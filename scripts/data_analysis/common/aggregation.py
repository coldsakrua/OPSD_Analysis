"""Aggregate per-token metrics into summary statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np


class MetricsAggregator:
    """Accumulate token-level JSD KL, top-k KL, log-ratio stats."""

    def __init__(self, topk_names: tuple[str, ...] = ("k1", "k16")) -> None:
        self.topk_names = topk_names
        self.n_tokens = 0
        self.n_rollouts = 0
        self.sum_jsd = 0.0
        self.sum_log_ratio = 0.0
        self.sum_topk: dict[str, float] = {k: 0.0 for k in topk_names}
        self.token_jsd: Counter[int] = Counter()
        self.token_jsd_sum: dict[int, float] = defaultdict(float)
        self.length_hist: Counter[str] = Counter()

    def update(self, metrics: dict[str, Any], *, window_label: str | None = None) -> None:
        n = metrics.get("n_tokens", 0)
        if n == 0:
            return
        self.n_rollouts += 1
        self.n_tokens += n
        jsd = metrics["jsd_kl"]
        lr = metrics["log_ratio_k1"]
        self.sum_jsd += float(sum(jsd))
        self.sum_log_ratio += float(sum(lr))
        for name in self.topk_names:
            if name == "k1":
                vals = lr
            else:
                vals = metrics.get("topk_kl", {}).get(name, [])
            self.sum_topk[name] += float(sum(vals))

        for tid, j in zip(metrics["token_ids"], jsd):
            tid = int(tid)
            self.token_jsd[tid] += 1
            self.token_jsd_sum[tid] += float(j)

        if window_label:
            self.length_hist[window_label] += n

    def top_loss_tokens(self, tokenizer: Any, k: int = 50) -> list[dict[str, Any]]:
        items = []
        for tid, cnt in self.token_jsd.most_common(k):
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
        # Re-sort by total JSD contribution (loss-dominant)
        return sorted(items, key=lambda x: x["sum_jsd_kl"], reverse=True)[:k]

    def summary(self, tokenizer: Any | None = None, top_token_k: int = 50) -> dict[str, Any]:
        n = max(self.n_tokens, 1)
        out: dict[str, Any] = {
            "n_rollouts": self.n_rollouts,
            "n_tokens": self.n_tokens,
            "mean_jsd_kl": self.sum_jsd / n,
            "mean_log_ratio_k1": self.sum_log_ratio / n,
            "mean_topk_kl": {name: self.sum_topk[name] / n for name in self.topk_names},
            "length_window_token_counts": dict(self.length_hist),
        }
        if tokenizer is not None:
            out["top_loss_dominant_tokens"] = self.top_loss_tokens(tokenizer, top_token_k)
        return out


def merge_summaries(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"groups": summaries}
