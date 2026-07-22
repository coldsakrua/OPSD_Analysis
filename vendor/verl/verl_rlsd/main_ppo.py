"""CAST wrapper entrypoint for veRL PPO training."""

from __future__ import annotations

import hydra
from omegaconf import OmegaConf


def run_ppo(config) -> None:
    import verl.trainer.main_ppo as verl_main

    from verl_rlsd.bootstrap import ensure_cast_patches

    class CastTaskRunner(verl_main.TaskRunner):
        def run(self, config):  # type: ignore[no-untyped-def]
            ensure_cast_patches()
            return super().run(config)

    verl_main.TaskRunner = CastTaskRunner
    verl_main.run_ppo(config)


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


if __name__ == "__main__":
    main()
