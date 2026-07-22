from setuptools import find_packages, setup

setup(
    name="verl_rlsd",
    version="0.0.0",
    packages=find_packages(),
    entry_points={
        "vllm.general_plugins": [
            "ministral_tokenizer = verl_rlsd.ministral_vllm_plugin:init_plugin",
        ],
    },
)
