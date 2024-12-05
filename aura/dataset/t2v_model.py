from typing import *
import random

from videosys import CogVideoXConfig, VideoSysEngine, OpenSoraConfig, LatteConfig
import torch.distributed as dist


STORAGE_PATH = "/disk/onyx-scratch/dullo009-fall2024"

class CogVideoXT2VideoPipeline():
    def __init__(self, **kwargs):
        config = CogVideoXConfig("THUDM/CogVideoX-2b", **kwargs)
        self.engine = VideoSysEngine(config)

    def __call__(self, prompt: str, steps: int = 49, path: str = "./video.mp4"):
        video = self.engine.generate(prompt, num_inference_steps=steps).video[0]
        self.engine.save_video(video, path)

class OpenSoraT2VideoPipeline():
    def __init__(self, **kwargs):
        config = OpenSoraConfig(**kwargs)
        self.engine = VideoSysEngine(config)
    
    def __call__(self, prompt: str, path: str = "./video.mp4", **kwargs):
        video = self.engine.generate(prompt, **kwargs).video[0]
        self.engine.save_video(video, path)

class LatteT2VideoPipeline():
    def __init__(self, **kwargs):
        config = LatteConfig("maxin-cn/Latte-1", **kwargs)
        self.engine = VideoSysEngine(config)
    
    def __call__(self, prompt: str, path: str = "./video.mp4", steps: int = 30, **kwargs):
        video = self.engine.generate(prompt, num_inference_steps = steps, **kwargs).video[0]
        self.engine.save_video(video, path)

REFIK_PROMPT = "Generate an abstract, large-scale digital artwork in a modern gallery space. The piece features flowing, organic forms with intricate textures of clusters, ribbons, and particles. Bold, contrasting colors dominate, creating depth, motion, and balance."

COLOR_INJECT = {
    "joy": "Bright yellows, oranges, and greens swirl smoothly, radiating playful energy.",
    "sadness": "Cool blues and muted purples flow in slow, melancholic waves, dotted with small, shimmering clusters that fade into shadowy depths.",
    "anger": "Fiery reds and deep blacks collide in jagged, turbulent forms, with bursts of orange and sparks of yellow erupting like molten lava.",
    "disgust": "Sickly greens and murky browns twist into chaotic, overlapping shapes, with globular particles oozing and clumping together in unsettling patterns.",
    "fear": "Dark grays and ominous blacks blend with piercing whites and sharp streaks of electric blue, forming tense, jagged swirls that evoke a sense of unease and urgency."
}

def create_prompt(emotion):
    if emotion not in COLOR_INJECT.keys():
        print(f"`{emotion}` it not a valid emotion. Using default `happy` emotion.")
        emotion = "happy"

    return " ".join([REFIK_PROMPT, COLOR_INJECT[emotion]])

if __name__ == "__main__":
    t2v_pipeline = LatteT2VideoPipeline(enable_pab=True)
    for emotion, color_prompt in COLOR_INJECT.items():
        print(f"Generating {emotion}")
        prompt = " ".join([REFIK_PROMPT, color_prompt])
        t2v_pipeline(prompt, f"output/refik_{emotion}.mp4")