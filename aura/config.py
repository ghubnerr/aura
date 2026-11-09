"""Runtime configuration for Aura.

Values are loaded from environment variables (and optional `.env`) so the same
code can run on a laptop browser client and a GPU host on the LAN.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_STORAGE = "./output"
_DEFAULT_SIGNALING_HOST = "127.0.0.1"
_DEFAULT_SIGNALING_PORT = 3030
_DEFAULT_BIND_IP = "127.0.0.1"
_DEFAULT_IVF_DIR = "./ivf_files"
_DEFAULT_EMOTION_EMBED_DIM = 2048
_DEFAULT_CLIP_PROJ_DIM = 128


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


@dataclass(frozen=True)
class AuraConfig:
    """Process-wide Aura settings."""

    storage_path: str = _DEFAULT_STORAGE
    signaling_host: str = _DEFAULT_SIGNALING_HOST
    signaling_port: int = _DEFAULT_SIGNALING_PORT
    bind_ip: str = _DEFAULT_BIND_IP
    ivf_dir: str = _DEFAULT_IVF_DIR
    emotion_embed_dim: int = _DEFAULT_EMOTION_EMBED_DIM
    clip_projection_dim: int = _DEFAULT_CLIP_PROJ_DIM
    openai_api_key: str | None = None
    gpu_host: str | None = None

    @property
    def aura_storage(self) -> str:
        return os.path.join(self.storage_path, "aura_storage")

    @property
    def emotion_dataset_dir(self) -> str:
        return os.path.join(self.aura_storage, "emotion_dataset")

    @property
    def emotion_checkpoint(self) -> str:
        return os.path.join(self.aura_storage, "aura_emotion_classifier.pth")

    @property
    def signaling_url(self) -> str:
        return f"ws://{self.signaling_host}:{self.signaling_port}/signaling"


def load_config() -> AuraConfig:
    """Load config from the environment."""
    storage = _env_str("STORAGE_PATH", _DEFAULT_STORAGE)
    os.makedirs(storage, exist_ok=True)
    os.environ.setdefault("STORAGE_PATH", storage)

    cfg = AuraConfig(
        storage_path=storage,
        signaling_host=_env_str("AURA_SIGNALING_HOST", _DEFAULT_SIGNALING_HOST),
        signaling_port=_env_int("AURA_SIGNALING_PORT", _DEFAULT_SIGNALING_PORT),
        bind_ip=_env_str("AURA_BIND_IP", _DEFAULT_BIND_IP),
        ivf_dir=_env_str("AURA_IVF_DIR", _DEFAULT_IVF_DIR),
        emotion_embed_dim=_env_int(
            "AURA_EMOTION_EMBED_DIM", _DEFAULT_EMOTION_EMBED_DIM
        ),
        clip_projection_dim=_env_int(
            "AURA_CLIP_PROJECTION_DIM", _DEFAULT_CLIP_PROJ_DIM
        ),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        gpu_host=os.environ.get("AURA_GPU_HOST"),
    )
    os.makedirs(cfg.aura_storage, exist_ok=True)
    os.makedirs(cfg.ivf_dir, exist_ok=True)
    return cfg


# Eager singleton used by modules that previously read env ad hoc.
CONFIG = load_config()
