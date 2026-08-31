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

    # Full-param, no reference / KL.
    config.actor_rollout_ref.model.lora_rank = 0
    config.actor_rollout_ref.actor.use_kl_loss = False
    config.algorithm.use_kl_in_reward = False
    config.actor_rollout_ref.hybrid_engine = True

    overlong_on = bool(OmegaConf.select(kwargs, "overlong_buffer_cfg.enable"))
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
        "[opsd-grpo] metrics: log critic/acc/mean (+ training/acc) from math_dapo reward_extra_info.",
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
                }
            },
            num_cpus=config.ray_init.num_cpus,
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

        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
        from verl.trainer.ppo.reward import load_reward_manager
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

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
