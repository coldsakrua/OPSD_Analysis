#!/usr/bin/env python3
"""Compare Qwen3-4B think vs no-think: generation + hidden/FFN activations.

Pipeline
--------
1) Sample N problems from OpenThoughts parquet.
2) Optional vLLM generation under both chat-template modes (max_new_tokens).
3) HF forward on paired prompts: layer-wise last-token hidden & FFN intermediates,
   plus a shared-prefix control (positions before the templates diverge).

Outputs under --output-dir:
  generations_{think,nothink}.jsonl
  activation_stats.json
  summary.json
  figures/*.png
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


USER_TEMPLATE = (
    "Problem: {problem}\n\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        default="/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b",
    )
    p.add_argument(
        "--dataset-path",
        default=(
            "/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/"
            "data/openthoughts/train-00000-of-00002.parquet"
        ),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4, help="HF activation batch size")
    p.add_argument("--gen-batch-size", type=int, default=32, help="vLLM continuous-batch hint")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-activations", action="store_true")
    p.add_argument("--activation-only-n", type=int, default=0, help="0 = all samples")
    return p.parse_args()


def build_user(problem: str) -> str:
    # Avoid str.format: problems often contain LaTeX braces like \frac{a}{b}.
    return (
        "Problem: "
        + problem.strip()
        + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    )


def make_prompt(tokenizer: Any, problem: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": build_user(problem)}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(enable_thinking),
    )


def shared_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def load_problems(path: str, n: int, seed: int, tokenizer: Any, max_prompt_length: int) -> list[dict]:
    df = pd.read_parquet(path, columns=["problem"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    out: list[dict] = []
    for idx in order:
        problem = str(df.iloc[int(idx)]["problem"]).strip()
        if not problem:
            continue
        pt = make_prompt(tokenizer, problem, True)
        pn = make_prompt(tokenizer, problem, False)
        ids_t = tokenizer(pt, add_special_tokens=False)["input_ids"]
        ids_n = tokenizer(pn, add_special_tokens=False)["input_ids"]
        if max(len(ids_t), len(ids_n)) > max_prompt_length:
            continue
        out.append(
            {
                "row_id": int(idx),
                "problem": problem,
                "prompt_think": pt,
                "prompt_nothink": pn,
                "prompt_len_think": len(ids_t),
                "prompt_len_nothink": len(ids_n),
                "shared_prefix_len": shared_prefix_len(ids_t, ids_n),
            }
        )
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"Only found {len(out)} prompts <= {max_prompt_length}; need {n}")
    return out


def run_generation(
    model_path: str,
    samples: list[dict],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_new_tokens + 64,
        disable_custom_all_reduce=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    summary: dict[str, Any] = {}
    for mode, key in [("think", "prompt_think"), ("nothink", "prompt_nothink")]:
        prompts = [s[key] for s in samples]
        t0 = time.time()
        print(f"[gen] mode={mode} n={len(prompts)} max_new_tokens={args.max_new_tokens}")
        outputs = llm.generate(prompts, sampling, use_tqdm=True)
        path = output_dir / f"generations_{mode}.jsonl"
        lengths = []
        with path.open("w", encoding="utf-8") as f:
            for s, o in zip(samples, outputs):
                text = o.outputs[0].text if o.outputs else ""
                n_tok = len(o.outputs[0].token_ids) if o.outputs else 0
                lengths.append(n_tok)
                rec = {
                    "row_id": s["row_id"],
                    "mode": mode,
                    "prompt_len": s[f"prompt_len_{mode}"],
                    "gen_tokens": n_tok,
                    "text": text,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        arr = np.asarray(lengths, dtype=np.float64)
        summary[mode] = {
            "n": int(len(arr)),
            "mean_gen_tokens": float(arr.mean()),
            "median_gen_tokens": float(np.median(arr)),
            "p90_gen_tokens": float(np.percentile(arr, 90)),
            "max_gen_tokens": float(arr.max()),
            "frac_hit_max": float((arr >= args.max_new_tokens - 1).mean()),
            "seconds": float(time.time() - t0),
            "path": str(path),
        }
        print(f"[gen] {mode} done: mean_len={summary[mode]['mean_gen_tokens']:.1f}")
    # free vLLM before HF load
    del llm
    torch.cuda.empty_cache()
    return summary


@torch.no_grad()
def run_activations(
    model_path: str,
    samples: list[dict],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    n_layers = int(model.config.num_hidden_layers)
    # Capture FFN intermediate = act(gate) * up, i.e. input to down_proj.
    ffn_last: dict[str, dict[int, torch.Tensor]] = {"think": {}, "nothink": {}}
    active = {"mode": "think"}

    hooks = []
    for i, layer in enumerate(model.model.layers):

        def make_pre_hook(layer_idx: int):
            def _hook(_module, inputs):
                x = inputs[0]
                ffn_last[active["mode"]][layer_idx] = x[:, -1, :].detach().float().cpu()

            return _hook

        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_pre_hook(i)))

    acc = {
        "hidden_cosine_last": np.zeros(n_layers + 1, dtype=np.float64),
        "hidden_rel_l2_last": np.zeros(n_layers + 1, dtype=np.float64),
        "hidden_cosine_shared": np.zeros(n_layers + 1, dtype=np.float64),
        "ffn_cosine_last": np.zeros(n_layers, dtype=np.float64),
        "ffn_rel_l2_last": np.zeros(n_layers, dtype=np.float64),
        "ffn_sparsity_think": np.zeros(n_layers, dtype=np.float64),
        "ffn_sparsity_nothink": np.zeros(n_layers, dtype=np.float64),
        "logit_kl_tn": 0.0,
        "logit_kl_nt": 0.0,
        "logit_js": 0.0,
        "logit_top1_agree": 0.0,
        "n": 0,
    }
    per_sample: dict[str, list[float]] = {
        "hidden_cosine_last_mean": [],
        "hidden_cosine_last_mid": [],
        "hidden_cosine_last_late": [],
        "ffn_cosine_last_mean": [],
        "ffn_cosine_last_late": [],
        "logit_js": [],
        "shared_prefix_len": [],
    }

    def rel_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        num = (a - b).norm(dim=-1)
        den = a.norm(dim=-1) + b.norm(dim=-1) + 1e-8
        return num / den

    def sparsity(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        return (x.abs() < eps).float().mean(dim=-1)

    n_use = len(samples) if args.activation_only_n <= 0 else min(args.activation_only_n, len(samples))
    subset = samples[:n_use]
    bs = args.batch_size

    for start in tqdm(range(0, n_use, bs), desc="activations"):
        batch = subset[start : start + bs]
        enc_t = tokenizer(
            [s["prompt_think"] for s in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        enc_n = tokenizer(
            [s["prompt_nothink"] for s in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)

        active["mode"] = "think"
        ffn_last["think"].clear()
        out_t = model(**enc_t, output_hidden_states=True, use_cache=False)
        hs_t = out_t.hidden_states
        logits_t = out_t.logits[:, -1, :].float()

        active["mode"] = "nothink"
        ffn_last["nothink"].clear()
        out_n = model(**enc_n, output_hidden_states=True, use_cache=False)
        hs_n = out_n.hidden_states
        logits_n = out_n.logits[:, -1, :].float()

        for bi, s in enumerate(batch):
            sp = int(s["shared_prefix_len"])
            per_sample["shared_prefix_len"].append(float(sp))

            hc_layers: list[float] = []
            for li in range(n_layers + 1):
                ht = hs_t[li][bi, -1, :].float().cpu()
                hn = hs_n[li][bi, -1, :].float().cpu()
                c = float(F.cosine_similarity(ht.unsqueeze(0), hn.unsqueeze(0)).item())
                r = float(rel_l2(ht.unsqueeze(0), hn.unsqueeze(0)).item())
                acc["hidden_cosine_last"][li] += c
                acc["hidden_rel_l2_last"][li] += r
                hc_layers.append(c)

                lt = int(enc_t["attention_mask"][bi].sum().item())
                ln = int(enc_n["attention_mask"][bi].sum().item())
                pad_t = hs_t[li].shape[1] - lt
                pad_n = hs_n[li].shape[1] - ln
                if sp > 0:
                    mt = hs_t[li][bi, pad_t : pad_t + sp, :].float().mean(dim=0).cpu()
                    mn = hs_n[li][bi, pad_n : pad_n + sp, :].float().mean(dim=0).cpu()
                    acc["hidden_cosine_shared"][li] += float(
                        F.cosine_similarity(mt.unsqueeze(0), mn.unsqueeze(0)).item()
                    )
                else:
                    acc["hidden_cosine_shared"][li] += 1.0

            fc_layers: list[float] = []
            for li in range(n_layers):
                ft = ffn_last["think"][li][bi]
                fn = ffn_last["nothink"][li][bi]
                c = float(F.cosine_similarity(ft.unsqueeze(0), fn.unsqueeze(0)).item())
                r = float(rel_l2(ft.unsqueeze(0), fn.unsqueeze(0)).item())
                acc["ffn_cosine_last"][li] += c
                acc["ffn_rel_l2_last"][li] += r
                acc["ffn_sparsity_think"][li] += float(sparsity(ft.unsqueeze(0)).item())
                acc["ffn_sparsity_nothink"][li] += float(sparsity(fn.unsqueeze(0)).item())
                fc_layers.append(c)

            # logits JS / KL
            lt_ = logits_t[bi]
            ln_ = logits_n[bi]
            log_pt = F.log_softmax(lt_, dim=-1)
            log_pn = F.log_softmax(ln_, dim=-1)
            pt = log_pt.exp()
            pn = log_pn.exp()
            kl_tn = float(F.kl_div(log_pn, pt, reduction="sum").item())
            kl_nt = float(F.kl_div(log_pt, pn, reduction="sum").item())
            m = 0.5 * (pt + pn)
            js = 0.5 * (
                float(F.kl_div(m.clamp_min(1e-12).log(), pt, reduction="sum").item())
                + float(F.kl_div(m.clamp_min(1e-12).log(), pn, reduction="sum").item())
            )
            top1_agree = float(lt_.argmax().item() == ln_.argmax().item())
            acc["logit_kl_tn"] += kl_tn
            acc["logit_kl_nt"] += kl_nt
            acc["logit_js"] += js
            acc["logit_top1_agree"] += top1_agree
            acc["n"] += 1

            mid = n_layers // 2
            per_sample["hidden_cosine_last_mean"].append(float(np.mean(hc_layers)))
            per_sample["hidden_cosine_last_mid"].append(hc_layers[mid])
            per_sample["hidden_cosine_last_late"].append(float(np.mean(hc_layers[-6:])))
            per_sample["ffn_cosine_last_mean"].append(float(np.mean(fc_layers)))
            per_sample["ffn_cosine_last_late"].append(float(np.mean(fc_layers[-6:])))
            per_sample["logit_js"].append(js)

        del out_t, out_n, hs_t, hs_n, logits_t, logits_n
        torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    n = max(acc["n"], 1)
    layer_stats = {
        "n_samples": acc["n"],
        "n_layers": n_layers,
        "hidden_cosine_last": (acc["hidden_cosine_last"] / n).tolist(),
        "hidden_rel_l2_last": (acc["hidden_rel_l2_last"] / n).tolist(),
        "hidden_cosine_shared_prefix": (acc["hidden_cosine_shared"] / n).tolist(),
        "ffn_cosine_last": (acc["ffn_cosine_last"] / n).tolist(),
        "ffn_rel_l2_last": (acc["ffn_rel_l2_last"] / n).tolist(),
        "ffn_sparsity_think": (acc["ffn_sparsity_think"] / n).tolist(),
        "ffn_sparsity_nothink": (acc["ffn_sparsity_nothink"] / n).tolist(),
        "logit_kl_think_to_nothink": acc["logit_kl_tn"] / n,
        "logit_kl_nothink_to_think": acc["logit_kl_nt"] / n,
        "logit_js_divergence": acc["logit_js"] / n,
        "logit_top1_agree_rate": acc["logit_top1_agree"] / n,
        "per_sample_summary": {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "median": float(np.median(v)),
                "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)),
            }
            for k, v in per_sample.items()
            if v
        },
    }

    # Save raw per-sample for later plots
    np.savez_compressed(
        output_dir / "activation_per_sample.npz",
        **{k: np.asarray(v, dtype=np.float64) for k, v in per_sample.items()},
        hidden_cosine_last_by_layer=acc["hidden_cosine_last"] / n,
        ffn_cosine_last_by_layer=acc["ffn_cosine_last"] / n,
        hidden_cosine_shared_by_layer=acc["hidden_cosine_shared"] / n,
        hidden_rel_l2_last_by_layer=acc["hidden_rel_l2_last"] / n,
        ffn_rel_l2_last_by_layer=acc["ffn_rel_l2_last"] / n,
    )

    with (output_dir / "activation_stats.json").open("w", encoding="utf-8") as f:
        json.dump(layer_stats, f, indent=2, ensure_ascii=False)

    make_figures(layer_stats, per_sample, output_dir / "figures")
    del model
    torch.cuda.empty_cache()
    return layer_stats


def make_figures(stats: dict, per_sample: dict, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable: {exc}")
        return

    hc = np.asarray(stats["hidden_cosine_last"])
    hs = np.asarray(stats["hidden_cosine_shared_prefix"])
    fc = np.asarray(stats["ffn_cosine_last"])
    hr = np.asarray(stats["hidden_rel_l2_last"])
    fr = np.asarray(stats["ffn_rel_l2_last"])

    # 1) layer-wise cosine
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(range(len(hc)), hc, label="last-token hidden cos(think, nothink)", lw=2)
    ax.plot(range(len(hs)), hs, label="shared-prefix mean hidden cos (control)", lw=2, ls="--")
    ax.plot(range(len(fc)), fc, label="last-token FFN intermediate cos", lw=2)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Think vs No-Think representation similarity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "layer_cosine.png", dpi=160)
    plt.close(fig)

    # 2) relative L2
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(range(len(hr)), hr, label="hidden relative L2", lw=2)
    ax.plot(range(len(fr)), fr, label="FFN relative L2", lw=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\|a-b\|_2 / (\|a\|_2+\|b\|_2)$")
    ax.set_title("Think vs No-Think relative L2 distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "layer_rel_l2.png", dpi=160)
    plt.close(fig)

    # 3) per-sample late-layer cosine hist
    if per_sample.get("hidden_cosine_last_late"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(per_sample["hidden_cosine_last_late"], bins=40, color="#1f77b4", alpha=0.85)
        axes[0].set_title("Late-layer hidden cosine (mean of last 6)")
        axes[0].set_xlabel("cosine")
        axes[1].hist(per_sample["ffn_cosine_last_late"], bins=40, color="#d62728", alpha=0.85)
        axes[1].set_title("Late-layer FFN cosine (mean of last 6)")
        axes[1].set_xlabel("cosine")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "per_sample_late_cosine_hist.png", dpi=160)
        plt.close(fig)

    if per_sample.get("logit_js"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(per_sample["logit_js"], bins=40, color="#2ca02c", alpha=0.85)
        ax.set_title("Next-token JS divergence (think vs nothink)")
        ax.set_xlabel("JS")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "logit_js_hist.png", dpi=160)
        plt.close(fig)


def build_verdict(act: dict | None, gen: dict | None) -> dict[str, Any]:
    verdict: dict[str, Any] = {"claim": "think vs no-think induce large internal differences"}
    if act:
        hc = np.asarray(act["hidden_cosine_last"])
        hs = np.asarray(act["hidden_cosine_shared_prefix"])
        fc = np.asarray(act["ffn_cosine_last"])
        late_h = float(hc[-6:].mean())
        late_f = float(fc[-6:].mean())
        early_shared = float(hs[:8].mean())
        verdict["activation"] = {
            "shared_prefix_early_cosine": early_shared,
            "late_hidden_cosine": late_h,
            "late_ffn_cosine": late_f,
            "logit_js": act["logit_js_divergence"],
            "logit_top1_agree": act["logit_top1_agree_rate"],
            "min_hidden_cosine_layer": int(hc.argmin()),
            "min_hidden_cosine": float(hc.min()),
            "min_ffn_cosine_layer": int(fc.argmin()),
            "min_ffn_cosine": float(fc.min()),
        }
        # Heuristic "large difference" criteria
        verdict["activation"]["large_difference"] = bool(
            late_h < 0.5
            and late_f < 0.3
            and early_shared > 0.95
            and (
                act["logit_js_divergence"] > 0.5
                or act["logit_top1_agree_rate"] < 0.05
            )
        )
    if gen:
        verdict["generation"] = gen
    return verdict


def load_or_prepare_samples(args: argparse.Namespace, output_dir: Path) -> list[dict]:
    samples_path = output_dir / "samples.jsonl"
    if samples_path.exists():
        samples = [
            json.loads(line)
            for line in samples_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(samples) >= args.num_samples:
            print(f"[init] reusing {args.num_samples} cached samples from {samples_path}")
            return samples[: args.num_samples]
        print(
            f"[init] cached samples={len(samples)} < requested={args.num_samples}; resampling"
        )

    print("[init] loading tokenizer / sampling problems")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    samples = load_problems(
        args.dataset_path, args.num_samples, args.seed, tokenizer, args.max_prompt_length
    )
    with samples_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    meta_path = output_dir / "samples_meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(
                json.dumps(
                    {
                        "row_id": s["row_id"],
                        "prompt_len_think": s["prompt_len_think"],
                        "prompt_len_nothink": s["prompt_len_nothink"],
                        "shared_prefix_len": s["shared_prefix_len"],
                        "problem_preview": s["problem"][:200],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[init] prepared {len(samples)} samples -> {samples_path}")
    return samples


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    samples = load_or_prepare_samples(args, output_dir)

    gen_summary = None
    if not args.skip_generate:
        gen_summary = run_generation(args.model_path, samples, output_dir, args)
        with (output_dir / "generation_summary.json").open("w", encoding="utf-8") as f:
            json.dump(gen_summary, f, indent=2)
    elif (output_dir / "generation_summary.json").exists():
        gen_summary = json.loads((output_dir / "generation_summary.json").read_text(encoding="utf-8"))

    act_stats = None
    if not args.skip_activations:
        act_stats = run_activations(args.model_path, samples, output_dir, args)
    elif (output_dir / "activation_stats.json").exists():
        act_stats = json.loads((output_dir / "activation_stats.json").read_text(encoding="utf-8"))

    summary = {
        "num_samples": len(samples),
        "generation": gen_summary,
        "activation": {
            k: act_stats[k]
            for k in [
                "n_samples",
                "logit_kl_think_to_nothink",
                "logit_kl_nothink_to_think",
                "logit_js_divergence",
                "logit_top1_agree_rate",
                "per_sample_summary",
            ]
        }
        if act_stats
        else None,
        "verdict": build_verdict(act_stats, gen_summary),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary["verdict"], indent=2, ensure_ascii=False))
    print(f"[done] outputs in {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
