#!/usr/bin/env python3
"""Full-param GRPO (DAPO clip-higher + token-mean) on top of installed verl.

This verl build only supports hybrid_engine (actor+vLLM colocated on the same GPUs).
ROLLOUT_GPUS / TRAIN_GPUS therefore encode the *intended* split for memory budgeting
(gpu_memory_utilization) while FSDP+vLLM sleep/wake share all GPUs — maximizing
utilization and syncing rollout weights every training step.

Env (optional, also overridable via hydra ++opsd.*):
  ROLLOUT_GPUS, TRAIN_GPUS  — must sum to trainer.n_gpus_per_node
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import hydra
import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf


def _intended_split(config: DictConfig) -> tuple[int, int]:
    n_gpus = int(config.trainer.n_gpus_per_node)
    opsd = OmegaConf.select(config, "opsd") or {}
    rollout = int(opsd.get("rollout_gpus") or os.environ.get("ROLLOUT_GPUS") or max(1, n_gpus // 2))
    train = int(opsd.get("train_gpus") or os.environ.get("TRAIN_GPUS") or (n_gpus - rollout))
    if rollout + train != n_gpus:
        raise ValueError(
            f"ROLLOUT_GPUS({rollout})+TRAIN_GPUS({train}) must equal n_gpus_per_node({n_gpus}). "
            "This verl build colocates actor+vLLM on all GPUs (hybrid_engine); "
            "the split only budgets vLLM gpu_memory_utilization."
        )
    if rollout < 1 or train < 1:
        raise ValueError(f"need at least 1 rollout and 1 train GPU, got {rollout}+{train}")
    return rollout, train


def _patch_vllm_listconfig_coercion() -> None:
    """OmegaConf ListConfig is list-like but fails vLLM `isinstance(..., list)`."""
    from vllm.sampling_params import SamplingParams

    if getattr(SamplingParams, "_opsd_listconfig_patched", False):
        return

    _orig = SamplingParams._verify_args

    def _verify(self):  # noqa: ANN001
        ids = getattr(self, "stop_token_ids", None)
        if ids is not None and not isinstance(ids, list):
            self.stop_token_ids = [int(x) for x in ids]
        stop = getattr(self, "stop", None)
        if stop is not None and not isinstance(stop, list):
            self.stop = list(stop)
        return _orig(self)

    SamplingParams._verify_args = _verify  # type: ignore[method-assign]
    SamplingParams._opsd_listconfig_patched = True  # type: ignore[attr-defined]


def _shield_cpu_actor_cuda_probes() -> None:
    """TaskRunner is num_gpus=0: Ray may set CUDA_VISIBLE_DEVICES=\"\".

    On this cluster torch can still report is_available=True with device_count=0, so
    transformers→torchao import-time CUDA probes crash. Force probes off for this
    process only. Under RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 this is usually
    a no-op (device_count>0).
    """
    import torch

    if int(torch.cuda.device_count()) != 0:
        return

    torch.cuda.is_available = lambda: False  # type: ignore[method-assign]
    torch.cuda.device_count = lambda: 0  # type: ignore[method-assign]
    torch.cuda.get_device_capability = lambda device=None: (0, 0)  # type: ignore[method-assign]
    try:
        import torch.utils._triton as _triton_utils

        _triton_utils.has_triton = lambda: False  # type: ignore[assignment]
    except Exception:
        pass


def _install_lazy_vllm_listconfig_patch() -> None:
    """Ray worker_process_setup_hook entry.

    With RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1, importing vllm here is safe:
    CUDA may init with full CVD; verl Worker later calls set_device(RAY_LOCAL_RANK).
    Keep an import hook as fallback if vllm is not yet importable at hook time.
    """
    import importlib
    import sys

    if getattr(sys, "_opsd_lazy_listconfig_hook", False):
        return
    sys._opsd_lazy_listconfig_hook = True  # type: ignore[attr-defined]

    try:
        _patch_vllm_listconfig_coercion()
    except Exception:
        pass

    _orig_import_module = importlib.import_module

    def _import_module(name, package=None):  # noqa: ANN001
        mod = _orig_import_module(name, package=package)
        if name == "vllm.sampling_params" or name == "vllm" or (
            isinstance(name, str) and name.startswith("vllm.")
        ):
            try:
                _patch_vllm_listconfig_coercion()
            except Exception:
                pass
        return mod

    importlib.import_module = _import_module  # type: ignore[assignment]


def _apply_opsd_defaults(config: DictConfig) -> None:
    """Fill DAPO/GRPO knobs if missing; tune vLLM util from intended split."""
    rollout_gpus, train_gpus = _intended_split(config)
    OmegaConf.set_struct(config, False)
    if OmegaConf.select(config, "opsd") is None:
        config.opsd = OmegaConf.create({})
    config.opsd.rollout_gpus = rollout_gpus
    config.opsd.train_gpus = train_gpus
    config.opsd.note = (
        "hybrid_engine colocates actor+vLLM on all n_gpus; "
        "rollout_gpus/train_gpus budget gpu_memory_utilization only"
    )

    # Prefer not to overwrite an explicit util from CLI.
    util = OmegaConf.select(config, "actor_rollout_ref.rollout.gpu_memory_utilization")
    if util is None or float(util) <= 0:
        # More intended rollout share → slightly higher KV reservation while awake.
        config.actor_rollout_ref.rollout.gpu_memory_utilization = round(
            min(0.75, 0.35 + 0.12 * rollout_gpus), 2
        )

    rm = config.reward_model
    rm.reward_manager = rm.get("reward_manager") or "dapo"
    kwargs = rm.get("reward_kwargs")
    if kwargs is None:
        kwargs = OmegaConf.create({})
        rm.reward_kwargs = kwargs
    max_resp = int(config.data.max_response_length)
    if OmegaConf.select(kwargs, "max_resp_len") is None:
        kwargs.max_resp_len = max_resp
    # Always pass a cfg object: DAPORewardManager accesses `.enable` unconditionally.
    buf = OmegaConf.select(kwargs, "overlong_buffer_cfg")
    if buf is None:
        kwargs.overlong_buffer_cfg = OmegaConf.create(
            {
                "enable": False,
                "len": min(4096, max(1, max_resp - 8192)),
                "penalty_factor": 1.0,
                "log": False,
            }
        )
    else:
        # Keep lengths if set, but default to disabled unless caller forces enable=true.
        if OmegaConf.select(buf, "enable") is None:
            buf.enable = False

    # Boxed-first reward (Qwen SFT rarely emits Minerva ``Answer:`` alone).
    reward_fn = OmegaConf.select(config, "custom_reward_function") or OmegaConf.create({})
    if OmegaConf.select(reward_fn, "path") in (None, ""):
        reward_path = str(Path(__file__).resolve().parent / "reward_math_dapo_boxed.py")
        config.custom_reward_function = OmegaConf.create(
            {"path": reward_path, "name": "compute_score"}
        )
    elif OmegaConf.select(config, "custom_reward_function") is None:
        config.custom_reward_function = reward_fn

    # Full-param, no reference / KL.
    config.actor_rollout_ref.model.lora_rank = 0
    config.actor_rollout_ref.actor.use_kl_loss = False
    config.algorithm.use_kl_in_reward = False
    config.actor_rollout_ref.hybrid_engine = True

    # Ensure vLLM stops on chat EOS (<|im_end|>) as well as <|endoftext|>.
    # SFT/base generation_config often only lists the latter → length clip floods.
    existing_stop = OmegaConf.select(config, "actor_rollout_ref.rollout.stop_token_ids")
    if not existing_stop:
        try:
            from transformers import AutoTokenizer

            from stop_tokens import resolve_stop_token_ids

            model_path = str(config.actor_rollout_ref.model.path)
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            stop_ids = resolve_stop_token_ids(tok)
            if stop_ids:
                config.actor_rollout_ref.rollout.stop_token_ids = list(stop_ids)
                print(f"[opsd-grpo] rollout.stop_token_ids={stop_ids}", flush=True)
            else:
                print("[opsd-grpo][WARN] could not resolve stop_token_ids", flush=True)
        except Exception as exc:
            print(f"[opsd-grpo][WARN] failed to set stop_token_ids: {exc}", flush=True)

    overlong_on = bool(OmegaConf.select(kwargs, "overlong_buffer_cfg.enable"))
    reward_path = OmegaConf.select(config, "custom_reward_function.path")
    print(
        f"[opsd-grpo] intended_split=rollout{rollout_gpus}+train{train_gpus} "
        f"n_gpus={config.trainer.n_gpus_per_node} "
        f"vllm_util={config.actor_rollout_ref.rollout.gpu_memory_utilization} "
        f"hybrid=1 free_cache_engine={config.actor_rollout_ref.rollout.free_cache_engine} "
        f"overlong_penalty={int(overlong_on)}",
        flush=True,
    )
    print(
        "[opsd-grpo] weight sync: each generate_sequences enters FSDPVLLMShardingManager "
        "→ state_dict/update_params into vLLM, then sleep+empty_cache on exit "
        "(policy update then next-step rollout sees new weights).",
        flush=True,
    )
    print(
        f"[opsd-grpo] reward: custom={reward_path} "
        "(boxed-first + Answer: fallback; critic/acc from reward_extra_info).",
        flush=True,
    )
    dump_n = int(OmegaConf.select(config, "opsd.rollout_dump_n") or 0)
    dump_dir = OmegaConf.select(config, "trainer.rollout_data_dir")
    print(
        f"[opsd-grpo] rollout dump: dir={dump_dir} max_per_step={dump_n or 'all'}",
        flush=True,
    )


def run_ppo(config: DictConfig) -> None:
    _apply_opsd_defaults(config)

    # Slurm/ROCm nodes often export both CUDA_* and ROCR_*; verl workers abort on that.
    for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        os.environ.pop(key, None)

    if not ray.is_initialized():
        # Disable dashboard: MetricsHead often dies under cluster nproc soft-limit (EOF),
        # which then stalls plasma_store and times out node startup.
        src_dir = str(Path(__file__).resolve().parent)
        py_path = os.environ.get("PYTHONPATH", "")
        py_path = f"{src_dir}:{py_path}" if py_path else src_dir
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
                    "VLLM_LOGGING_LEVEL": os.environ.get("VLLM_LOGGING_LEVEL", "WARN"),
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                    # Explicitly clear so Ray workers do not inherit Slurm ROCR/HIP.
                    "ROCR_VISIBLE_DEVICES": "",
                    "HIP_VISIBLE_DEVICES": "",
                    # verl path: keep full CVD, Worker.set_device(RAY_LOCAL_RANK).
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": os.environ.get(
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1"
                    ),
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
                    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
                    "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "1"),
                    "PYTHONPATH": py_path,
                },
                # Lazy: must not import vllm here (would init CUDA before per-worker CVD).
                "worker_process_setup_hook": "train_grpo_dapo._install_lazy_vllm_listconfig_patch",
            },
            num_cpus=config.ray_init.num_cpus,
            include_dashboard=False,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


def _patch_compute_data_metrics_with_acc() -> None:
    """verl's compute_data_metrics omits reward_extra_info['acc']; add mean acc to step logs."""
    import verl.trainer.ppo.metric_utils as metric_utils
    import verl.trainer.ppo.ray_trainer as ray_trainer

    if getattr(metric_utils.compute_data_metrics, "_opsd_acc_patched", False):
        return

    _orig = metric_utils.compute_data_metrics

    def _wrapped(batch, use_critic: bool = True):
        metrics = _orig(batch, use_critic=use_critic)
        ntb = getattr(batch, "non_tensor_batch", None) or {}
        if "acc" in ntb:
            acc = np.asarray(ntb["acc"], dtype=np.float64).reshape(-1)
            if acc.size:
                mean_acc = float(acc.mean())
                metrics["critic/acc/mean"] = mean_acc
                metrics["training/acc"] = mean_acc
        return metrics

    _wrapped._opsd_acc_patched = True  # type: ignore[attr-defined]
    metric_utils.compute_data_metrics = _wrapped
    ray_trainer.compute_data_metrics = _wrapped


def _patch_rollout_dump_subsample(config: DictConfig) -> None:
    """Keep only up to opsd.rollout_dump_n samples when writing trainer.rollout_data_dir JSONL."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if getattr(RayPPOTrainer._dump_generations, "_opsd_subsample_patched", False):
        return

    max_n = int(OmegaConf.select(config, "opsd.rollout_dump_n") or 0)
    _orig = RayPPOTrainer._dump_generations

    def _wrapped(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        n = len(inputs)
        keep = max_n if max_n > 0 else n
        if keep < n:
            # Prefer a mix of high/low scores so all-wrong vs rare-correct is visible.
            order = sorted(range(n), key=lambda i: float(scores[i]), reverse=True)
            half = max(1, keep // 2)
            chosen = order[:half] + order[-(keep - half) :]
            # Stable unique, preserve score extremes first.
            seen = set()
            idx = []
            for i in chosen:
                if i not in seen:
                    seen.add(i)
                    idx.append(i)
            if len(idx) < keep:
                for i in range(n):
                    if i not in seen:
                        idx.append(i)
                        seen.add(i)
                    if len(idx) >= keep:
                        break
            inputs = [inputs[i] for i in idx]
            outputs = [outputs[i] for i in idx]
            scores = [scores[i] for i in idx]
            reward_extra_infos_dict = {
                k: [v[i] for i in idx] if isinstance(v, (list, tuple)) and len(v) == n else v
                for k, v in (reward_extra_infos_dict or {}).items()
            }
            print(
                f"[opsd-grpo] rollout dump subsample {n} -> {len(inputs)} (max_n={max_n})",
                flush=True,
            )
        return _orig(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path)

    _wrapped._opsd_subsample_patched = True  # type: ignore[attr-defined]
    RayPPOTrainer._dump_generations = _wrapped


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config: DictConfig):
        from pprint import pprint

        # Protect against empty-CVD torchao import crashes when Ray clears GPUs.
        _shield_cpu_actor_cuda_probes()

        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
        from verl.trainer.ppo.reward import load_reward_manager
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        _patch_vllm_listconfig_coercion()
        _patch_compute_data_metrics_with_acc()
        _patch_rollout_dump_subsample(config)

        for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
            os.environ.pop(key, None)

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}", flush=True)
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Overlay chat template from instruct / dedicated path when training a base ckpt.
        chat_template_path = OmegaConf.select(config, "opsd.chat_template_path")
        if chat_template_path:
            overlay = hf_tokenizer(str(chat_template_path), trust_remote_code=trust_remote_code)
            if getattr(overlay, "chat_template", None):
                tokenizer.chat_template = overlay.chat_template
                print(f"[opsd-grpo] overlay chat_template from {chat_template_path}", flush=True)
        custom_tpl = OmegaConf.select(config, "actor_rollout_ref.model.custom_chat_template")
        if custom_tpl:
            tokenizer.chat_template = custom_tpl

        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError(config.actor_rollout_ref.actor.strategy)

        # Single global pool: hybrid actor+rollout on all GPUs (only mode supported).
        # Intended ROLLOUT/TRAIN split is recorded on config.opsd for logs / util budget.
        n_gpus = int(config.trainer.n_gpus_per_node)
        nnodes = int(config.trainer.nnodes)
        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [n_gpus] * nnodes}
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Explicitly skip RefPolicy (no KL / no reference model).
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            raise RuntimeError("opsd GRPO DAPO entry forbids reference/KL; set use_kl_loss=false")

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Keep nested cfgs as OmegaConf so DAPORewardManager can use attr access
        # (e.g. overlong_buffer_cfg.enable), not plain dicts.
        raw_kwargs = config.reward_model.get("reward_kwargs", {}) or {}
        reward_kwargs = OmegaConf.to_container(OmegaConf.create(raw_kwargs), resolve=True)
        if isinstance(reward_kwargs.get("overlong_buffer_cfg"), dict):
            reward_kwargs["overlong_buffer_cfg"] = OmegaConf.create(reward_kwargs["overlong_buffer_cfg"])
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **reward_kwargs)
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **reward_kwargs)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


# Resolve verl's packaged hydra config directory.
_VERL_CONFIG_DIR = str(Path(__import__("verl").__file__).resolve().parent / "trainer" / "config")


@hydra.main(config_path=_VERL_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config: DictConfig) -> None:
    run_ppo(config)


if __name__ == "__main__":
    main()
