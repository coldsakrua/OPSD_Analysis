"""Unit smoke: global_steps wrap must attach after init_workers, not only __init__."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

# Allow `python -m` / direct path execution without installing the package.
_VERL_ROOT = Path(__file__).resolve().parents[2]
if str(_VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERL_ROOT))


def test_wrap_runs_after_init_workers() -> None:
    # Build a fake verl.trainer.ppo.ray_trainer module before importing the patcher.
    fake_mod = types.ModuleType("verl.trainer.ppo.ray_trainer")

    class FakeRayPPOTrainer:
        def __init__(self):
            self.global_steps = 0
            self.config = {}  # TeacherEMAController.from_trainer reads this
            # Intentionally missing actor_rollout_wg until init_workers().

        def init_workers(self):
            def compute_log_prob(batch):
                return "ok_compute"

            def update_actor(batch):
                return "ok_update"

            self.actor_rollout_wg = SimpleNamespace(
                compute_log_prob=compute_log_prob,
                update_actor=update_actor,
            )

    fake_mod.RayPPOTrainer = FakeRayPPOTrainer
    sys.modules["verl.trainer.ppo.ray_trainer"] = fake_mod
    for name in ("verl", "verl.trainer", "verl.trainer.ppo"):
        sys.modules.setdefault(name, types.ModuleType(name))

    from verl_rlsd import teacher_ema

    FakeRayPPOTrainer._rlsd_teacher_ema_patched = False  # type: ignore[attr-defined]
    teacher_ema._PATCHED = False
    teacher_ema._patch_ray_trainer_init()

    trainer = FakeRayPPOTrainer()
    assert getattr(trainer, "actor_rollout_wg", None) is None

    trainer.init_workers()
    wg = trainer.actor_rollout_wg
    assert getattr(wg, "_rlsd_global_steps_wrapped", False) is True

    batch = SimpleNamespace(meta_info={})
    trainer.global_steps = 50
    assert wg.compute_log_prob(batch) == "ok_compute"
    assert batch.meta_info["global_steps"] == 50

    batch2 = SimpleNamespace(meta_info={})
    trainer.global_steps = 100
    assert wg.update_actor(batch2) == "ok_update"
    assert batch2.meta_info["global_steps"] == 100

    print("PASS: global_steps wrap attaches after init_workers and injects step")


if __name__ == "__main__":
    test_wrap_runs_after_init_workers()
