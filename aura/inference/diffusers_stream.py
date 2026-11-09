"""Realtime diffusion backend: current frame + emotion -> next frame.

Uses Stable Diffusion Turbo img2img so generation stays temporally continuous
(the previous art frame is the init image). Emotion embedding selects an
abstract-art prompt; embedding energy modulates denoise strength.

Fits an RTX 5060 Ti 16GB. StreamDiffusionV2 / Krea need ~24GB+ and stay optional
behind ``AURA_ART_BACKEND=streamdiffusionv2`` once VRAM allows.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

# Abstract prompts — no faces/people; palette + motion keyed by emotion.
EMOTION_PROMPTS = {
    "happy": (
        "abstract ethereal art, warm golden amber light flowing in soft curves, "
        "luminous paint swirls, joyful atmosphere, no people, no face, no text"
    ),
    "sad": (
        "abstract ethereal art, deep blue indigo watercolor dissolving slowly, "
        "cool mist, melancholy calm, no people, no face, no text"
    ),
    "anger": (
        "abstract ethereal art, violent crimson and ember streaks, sharp jagged "
        "energy, turbulent contrast, no people, no face, no text"
    ),
    "fear": (
        "abstract ethereal art, dark violet shadows with uneasy flicker, "
        "fractured light, tense atmosphere, no people, no face, no text"
    ),
    "surprise": (
        "abstract ethereal art, sudden neon magenta and yellow bursts, "
        "electric sparks, explosive motion, no people, no face, no text"
    ),
    "disgust": (
        "abstract ethereal art, sickly green and olive textures, viscous "
        "organic forms, uneasy stillness, no people, no face, no text"
    ),
    "contempt": (
        "abstract ethereal art, cold muted teal and slate geometry, distant "
        "detached shapes, no people, no face, no text"
    ),
    "neutral": (
        "abstract ethereal art, soft gray graphite gradients, gentle ambient "
        "motion, balanced calm, no people, no face, no text"
    ),
}

NEGATIVE_PROMPT = (
    "photo, realistic, face, person, human, portrait, text, watermark, logo, "
    "cartoon character, low quality, blurry"
)


class DiffusersTurboBackend:
    """SD-Turbo img2img backend implementing continuous ``step(frame, embedding)``."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 2,
        base_strength: float = 0.42,
        device: Optional[str] = None,
    ) -> None:
        import torch

        self.model_id = model_id or os.environ.get(
            "AURA_DIFFUSION_MODEL", "stabilityai/sd-turbo"
        )
        self.width = width
        self.height = height
        self.steps = max(1, int(os.environ.get("AURA_DIFFUSION_STEPS", steps)))
        self.base_strength = float(
            os.environ.get("AURA_DIFFUSION_STRENGTH", base_strength)
        )
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._emotion_hint = "neutral"
        self._prompt = EMOTION_PROMPTS["neutral"]
        self._pipe = None
        self._torch = torch

    def _ensure_pipe(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        from diffusers import AutoPipelineForImage2Image

        print(f"DiffusersTurboBackend: loading {self.model_id} on {self.device}")
        kwargs = {
            "torch_dtype": self.dtype,
        }
        if self.device == "cuda":
            kwargs["variant"] = "fp16"

        pipe = AutoPipelineForImage2Image.from_pretrained(self.model_id, **kwargs)
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        # SD-Turbo is guidance-distilled; keep CFG off for speed.
        self._pipe = pipe
        print("DiffusersTurboBackend: ready")
        return pipe

    def set_emotion_hint(self, emotion: Optional[str]) -> None:
        if not emotion:
            return
        key = emotion.lower().strip()
        if key in EMOTION_PROMPTS:
            self._emotion_hint = key
            self._prompt = EMOTION_PROMPTS[key]

    def _strength_from_embedding(self, embedding: np.ndarray) -> float:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.size == 0:
            return self.base_strength
        energy = float(np.linalg.norm(emb) / max(np.sqrt(emb.size), 1.0))
        # Map typical unit-ish norms into a small band around base_strength.
        delta = float(np.clip((energy / 8.0) - 0.5, -0.12, 0.15))
        return float(np.clip(self.base_strength + delta, 0.28, 0.65))

    def step(
        self,
        current_frame: np.ndarray,
        embedding: np.ndarray,
        n_frames: int = 1,
    ) -> list[np.ndarray]:
        from PIL import Image

        pipe = self._ensure_pipe()
        strength = self._strength_from_embedding(embedding)

        if current_frame.dtype != np.uint8 or current_frame.ndim != 3:
            raise ValueError("current_frame must be uint8 HxWx3 RGB")

        image = Image.fromarray(current_frame, mode="RGB").resize(
            (self.width, self.height),
            Image.Resampling.LANCZOS,
        )

        out_frames: list[np.ndarray] = []
        for _ in range(max(1, n_frames)):
            result = pipe(
                prompt=self._prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=image,
                num_inference_steps=self.steps,
                strength=strength,
                guidance_scale=0.0,
            )
            pil = result.images[0]
            image = pil  # feed forward for multi-frame steps
            arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
            out_frames.append(arr)

        return out_frames


class StreamDiffusionBackend:
    """Optional StreamDiffusion (v1) wrapper if the package is installed.

    Same contract as DiffusersTurboBackend; prefers TensorRT when available.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        width: int = 512,
        height: int = 512,
    ) -> None:
        self.model_id = model_id or os.environ.get(
            "AURA_DIFFUSION_MODEL", "stabilityai/sd-turbo"
        )
        self.width = width
        self.height = height
        self._emotion_hint = "neutral"
        self._prompt = EMOTION_PROMPTS["neutral"]
        self._stream = None

    def set_emotion_hint(self, emotion: Optional[str]) -> None:
        if emotion and emotion in EMOTION_PROMPTS:
            self._emotion_hint = emotion
            self._prompt = EMOTION_PROMPTS[emotion]
            if self._stream is not None:
                try:
                    self._stream.update_prompt(self._prompt)
                except Exception:
                    self._stream.prepare(
                        prompt=self._prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                    )

    def _ensure(self):
        if self._stream is not None:
            return self._stream
        from streamdiffusion import StreamDiffusionWrapper

        acceleration = os.environ.get("AURA_STREAM_ACCEL", "none")
        print(
            f"StreamDiffusionBackend: loading {self.model_id} "
            f"(acceleration={acceleration})"
        )
        stream = StreamDiffusionWrapper(
            model_id_or_path=self.model_id,
            t_index_list=[0, 16, 32, 45],
            mode="img2img",
            frame_buffer_size=1,
            width=self.width,
            height=self.height,
            warmup=5,
            acceleration=acceleration,
            use_lcm_lora=False,
            use_tiny_vae=True,
            cfg_type="none",
            seed=2,
        )
        stream.prepare(prompt=self._prompt, negative_prompt=NEGATIVE_PROMPT)
        self._stream = stream
        print("StreamDiffusionBackend: ready")
        return stream

    def step(
        self,
        current_frame: np.ndarray,
        embedding: np.ndarray,
        n_frames: int = 1,
    ) -> list[np.ndarray]:
        from PIL import Image

        stream = self._ensure()
        # Embedding currently steers via emotion hint / prompt; reserved for
        # future IP-Adapter / custom cross-attn injection.
        _ = embedding

        image = Image.fromarray(current_frame, mode="RGB").resize(
            (self.width, self.height),
            Image.Resampling.LANCZOS,
        )
        out: list[np.ndarray] = []
        for _ in range(max(1, n_frames)):
            result = stream.img2img(image)
            if not isinstance(result, Image.Image):
                result = Image.fromarray(np.asarray(result))
            image = result
            out.append(np.asarray(result.convert("RGB"), dtype=np.uint8))
        return out


def create_art_backend(kind: Optional[str] = None):
    """Factory used by ContinuousArtEngine.

    ``AURA_ART_BACKEND`` values:
      - ``morphing`` (default): CPU stub
      - ``diffusers_turbo``: SD-Turbo img2img (recommended on 16GB)
      - ``streamdiffusion``: StreamDiffusion package if installed
      - ``auto``: try CUDA diffusers_turbo, else morphing
    """
    from aura.inference.continuous_art import MorphingStubBackend

    choice = (kind or os.environ.get("AURA_ART_BACKEND", "auto")).strip().lower()

    if choice in {"morphing", "stub"}:
        return MorphingStubBackend()

    if choice == "streamdiffusion":
        try:
            return StreamDiffusionBackend()
        except Exception as exc:  # noqa: BLE001
            print(f"StreamDiffusionBackend unavailable ({exc}); falling back")
            return MorphingStubBackend()

    if choice == "diffusers_turbo":
        try:
            return DiffusersTurboBackend()
        except Exception as exc:  # noqa: BLE001
            print(f"DiffusersTurboBackend unavailable ({exc}); falling back")
            return MorphingStubBackend()

    if choice == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                print("AURA_ART_BACKEND=auto -> DiffusersTurboBackend (CUDA)")
                return DiffusersTurboBackend()
        except Exception as exc:  # noqa: BLE001
            print(f"auto GPU backend failed ({exc})")
        print("AURA_ART_BACKEND=auto -> MorphingStubBackend")
        return MorphingStubBackend()

    print(f"Unknown AURA_ART_BACKEND={choice!r}; using morphing stub")
    return MorphingStubBackend()
