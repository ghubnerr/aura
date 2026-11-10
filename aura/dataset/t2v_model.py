"""Text-to-video backends used to synthesize Aura training pairs.

OpenSora remains available for legacy runs. Prefer Wan 2.1 / Diffusers-based
pipelines for new dataset generation.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

REFIK_PROMPT = (
    "Generate an abstract, large-scale digital artwork in a modern gallery "
    "space. The piece features flowing, organic forms with intricate textures "
    "of clusters, ribbons, and particles. Bold, contrasting colors dominate, "
    "creating depth, motion, and balance."
)

COLOR_INJECT: Mapping[str, str] = {
    "happy": (
        "Bright yellows, oranges, and greens swirl smoothly, radiating playful "
        "energy."
    ),
    "joy": (
        "Bright yellows, oranges, and greens swirl smoothly, radiating playful "
        "energy."
    ),
    "sad": (
        "Cool blues and muted purples flow in slow, melancholic waves, dotted "
        "with small, shimmering clusters that fade into shadowy depths."
    ),
    "sadness": (
        "Cool blues and muted purples flow in slow, melancholic waves, dotted "
        "with small, shimmering clusters that fade into shadowy depths."
    ),
    "anger": (
        "Fiery reds and deep blacks collide in jagged, turbulent forms, with "
        "bursts of orange and sparks of yellow erupting like molten lava."
    ),
    "disgust": (
        "Sickly greens and murky browns twist into chaotic, overlapping shapes, "
        "with globular particles oozing and clumping together in unsettling "
        "patterns."
    ),
    "fear": (
        "Dark grays and ominous blacks blend with piercing whites and sharp "
        "streaks of electric blue, forming tense, jagged swirls that evoke a "
        "sense of unease and urgency."
    ),
    "surprise": (
        "Sudden flashes of electric cyan and magenta burst through soft white "
        "haze, with rapid expanding rings and sharp particle sprays."
    ),
    "neutral": (
        "Balanced soft grays and muted teals drift in calm horizontal layers "
        "with gentle, even particle motion."
    ),
    "contempt": (
        "Asymmetric cool silvers and desaturated greens shear past each other "
        "in rigid planes with sparse, cutting lines."
    ),
}


def create_prompt(emotion: str) -> str:
    """Build a Refik-style abstract art prompt for ``emotion``."""
    key = emotion.lower().strip()
    if key not in COLOR_INJECT:
        print(f"`{emotion}` is not a valid emotion. Using default `happy`.")
        key = "happy"
    return " ".join([REFIK_PROMPT, COLOR_INJECT[key]])


class _VideoSysPipeline:
    """Thin wrapper around VideoSys engines when that dependency is installed."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def __call__(self, prompt: str, path: str = "./video.mp4", **kwargs: Any) -> str:
        video = self.engine.generate(prompt, **kwargs).video[0]
        self.engine.save_video(video, path)
        return path


class CogVideoXT2VideoPipeline(_VideoSysPipeline):
    def __init__(self, **kwargs: Any) -> None:
        from videosys import CogVideoXConfig, VideoSysEngine

        config = CogVideoXConfig("THUDM/CogVideoX-2b", **kwargs)
        super().__init__(VideoSysEngine(config))

    def __call__(
        self, prompt: str, steps: int = 49, path: str = "./video.mp4", **kwargs: Any
    ) -> str:
        video = self.engine.generate(prompt, num_inference_steps=steps, **kwargs).video[
            0
        ]
        self.engine.save_video(video, path)
        return path


class OpenSoraT2VideoPipeline(_VideoSysPipeline):
    def __init__(self, **kwargs: Any) -> None:
        from videosys import OpenSoraConfig, VideoSysEngine

        super().__init__(VideoSysEngine(OpenSoraConfig(**kwargs)))


class LatteT2VideoPipeline(_VideoSysPipeline):
    def __init__(self, **kwargs: Any) -> None:
        from videosys import LatteConfig, VideoSysEngine

        super().__init__(VideoSysEngine(LatteConfig("maxin-cn/Latte-1", **kwargs)))

    def __call__(
        self, prompt: str, path: str = "./video.mp4", steps: int = 30, **kwargs: Any
    ) -> str:
        video = self.engine.generate(
            prompt, num_inference_steps=steps, **kwargs
        ).video[0]
        self.engine.save_video(video, path)
        return path


class WanT2VideoPipeline:
    """Diffusers Wan 2.1 text-to-video pipeline (preferred synthetic data backend)."""

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        device: Optional[str] = None,
        dtype: Optional[str] = "bfloat16",
    ) -> None:
        import torch
        from diffusers import AutoencoderKLWan, WanPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import (
            UniPCMultistepScheduler,
        )

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype or "bfloat16", torch.bfloat16)

        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )
        self.pipe = WanPipeline.from_pretrained(
            model_id, vae=vae, torch_dtype=torch_dtype
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=3.0
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe.to(self.device)

    def __call__(
        self,
        prompt: str,
        path: str = "./video.mp4",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        steps: int = 30,
        guidance_scale: float = 5.0,
        **_kwargs: Any,
    ) -> str:
        from diffusers.utils import export_to_video

        result = self.pipe(
            prompt=prompt,
            negative_prompt=(
                "Bright tones, overexposed, static, blurred details, subtitles, "
                "style, works, paintings, images, static, overall gray, worst "
                "quality, low quality, JPEG compression residue, ugly, incomplete"
            ),
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
        )
        frames = result.frames[0]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        export_to_video(frames, path, fps=16)
        return path


def build_t2v_pipeline(name: str = "wan", **kwargs: Any):
    """Factory for dataset generation backends."""
    key = name.lower().strip()
    if key in {"wan", "wan2.1", "wan21"}:
        return WanT2VideoPipeline(**kwargs)
    if key in {"opensora", "open-sora"}:
        return OpenSoraT2VideoPipeline(**kwargs)
    if key == "latte":
        return LatteT2VideoPipeline(**kwargs)
    if key in {"cogvideox", "cogvideo"}:
        return CogVideoXT2VideoPipeline(**kwargs)
    raise ValueError(f"Unknown T2V backend: {name}")


if __name__ == "__main__":
    from aura.config import CONFIG

    pipeline = build_t2v_pipeline("wan")
    save_dir = os.path.join(CONFIG.storage_path, "video_dataset")
    os.makedirs(save_dir, exist_ok=True)
    for emotion, color_prompt in COLOR_INJECT.items():
        if emotion in {"joy", "sadness"}:
            # Skip aliases so we only generate one clip per palette.
            continue
        prompt = " ".join([REFIK_PROMPT, color_prompt])
        out = os.path.join(save_dir, f"refik_{emotion}.mp4")
        print(f"Generating {emotion} → {out}")
        pipeline(prompt, path=out)
