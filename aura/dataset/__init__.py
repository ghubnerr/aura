"""Dataset download, pair generation, and T2V backends.

Heavy optional deps (kagglehub, videosys, diffusers) are imported lazily from
the submodules that need them so WebRTC entrypoints stay lightweight.
"""

from aura.dataset.clip_provider import PairsProvider
from aura.dataset.t2v_model import COLOR_INJECT, create_prompt

__all__ = [
    "COLOR_INJECT",
    "PairsProvider",
    "create_prompt",
    "DatasetProvider",
    "PairsGenerator",
    "WanT2VideoPipeline",
    "OpenSoraT2VideoPipeline",
    "LatteT2VideoPipeline",
    "CogVideoXT2VideoPipeline",
    "build_t2v_pipeline",
]


def __getattr__(name: str):
    if name == "DatasetProvider":
        from aura.dataset.dataset_provider import DatasetProvider

        return DatasetProvider
    if name == "PairsGenerator":
        from aura.dataset.pair import PairsGenerator

        return PairsGenerator
    if name in {
        "WanT2VideoPipeline",
        "OpenSoraT2VideoPipeline",
        "LatteT2VideoPipeline",
        "CogVideoXT2VideoPipeline",
        "build_t2v_pipeline",
    }:
        from aura.dataset import t2v_model

        return getattr(t2v_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"
