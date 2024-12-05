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

PROMPT = "An abstract background featuring clean, crisp lines and symmetrical geometric patterns in harmonious colors moving from top to bottom. Use smooth gradients and organized shapes to convey neatness and meticulous care."

prompts = {
    "well-kept": "An abstract background featuring organized leaf vines and geometric patterns in harmious colors growing and moving through the piece.",
    "formal": "An abstract background featuring thick rectangular blocks and geometric patterns in harmious colors growing to consume the piece.",
    "simple": "An abstract background featuring simple, crisp lines and symmetrical geometric patterns in harmious colors moving from top to bottom.",
    "elegant": "An abstract background featuring elegant, thick marble etchings and symmetrical geometric patterns in harmious colors moving from top to bottom.",
    "dirty": "An abstract background featuring dirty, mishaped botches but in symmetrical geometric patterns all in harmious colors growing and shrinking.",
    "unorganized": "An abstract background featuring different shapes and lines and symmetrical geometric patterns in harmious colors moving from top to bottom.",
    "uncaring": "An abstract background featuring a single circle centered in the painting surrounded by harmious colors moving from top to bottom."
}
smooth_prompt = "Use smooth gradients and organized shapes to convey neatness and meticulous care."

color_prompt = "crisp lines should move quickly and have highlights of"
colors = {
    "happy": "gold and yellow",
    "sad": "blue and dark purple",
    "disgust": "green and burnt orange",
    "fear": "purple and grey white",
    "anger": "red and metallic orange",
    "neutral": "tan and white"
}

icons = {
    "well-kept": "organized vine shapes",
    "formal": "thick rectangular blocks",
    "simple": "crisp lines",
    "elegant": "thick marble etchings",
    "dirty": "mishaped blotches",
    "unorganized": "random shapes",
    "uncaring": "plain circle"
}

refik_prompt = "Generate an abstract, large-scale digital artwork in a modern gallery space. The piece features flowing, organic forms with intricate textures of clusters, ribbons, and particles. Bold, contrasting colors dominate, creating depth, motion, and balance."
refik_emotions = {
    "joy": "Bright yellows, oranges, and greens swirl smoothly, radiating playful energy.",
    "sadness": "Cool blues and muted purples flow in slow, melancholic waves, dotted with small, shimmering clusters that fade into shadowy depths.",
    "anger": "Fiery reds and deep blacks collide in jagged, turbulent forms, with bursts of orange and sparks of yellow erupting like molten lava.",
    "disgust": "Sickly greens and murky browns twist into chaotic, overlapping shapes, with globular particles oozing and clumping together in unsettling patterns.",
    "fear": "Dark grays and ominous blacks blend with piercing whites and sharp streaks of electric blue, forming tense, jagged swirls that evoke a sense of unease and urgency."
}

def create_prompt(presentation, emotion):
    if presentation not in prompts.keys():
        print(f"`{presentation}` it not a valid presentation. Using default `simple` presentation.")
        presentation = "simple"

    if emotion not in colors.keys():
        print(f"`{emotion}` it not a valid emotion. Using default `happy` emotion.")
        emotion = "happy"

    subject_prompt = f"{prompts[presentation]} f{smooth_prompt}."
    color_prompt = f"The {icons[presentation]} should move quickly have highlights of {colors[emotion]}."
    
    return " ".join([subject_prompt, color_prompt])

if __name__ == "__main__":
    emotions = list(refik_emotions.keys())
    random_indices = [random.randint(0, len(emotions) - 1) for _ in range(10_000)]
    count = 0
    for i in random_indices:
        count += 1
        emotion = emotions[i]
        prompt = " ".join([refik_prompt, refik_emotions[emotion]])
        print(f"Generating Video #{count}, Emotion: {emotion}")
        t2v_pipeline = LatteT2VideoPipeline(enable_pab=True, num_gpus=8)
        t2v_pipeline(prompt, f"{STORAGE_PATH}/backdrops/refik/refik_{emotion}_{count}.mp4")