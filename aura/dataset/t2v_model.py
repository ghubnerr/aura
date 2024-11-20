from typing import *
from videosys import CogVideoXConfig, VideoSysEngine, OpenSoraConfig
import torch.distributed as dist

STORAGE_PATH = "/a/buffalo.cs.fiu.edu./disk/jccl-002/homes/glucc002/Desktop/Projects/aura/dataset/temp"

class CogVideoXT2VideoPipeline():
    def __init__(self, **kwargs):
        config = CogVideoXConfig("THUDM/CogVideoX-2b", **kwargs)
        self.engine = VideoSysEngine(config)

    def __call__(self, prompt: str, steps: int, path: str = "./dataset/temp/video.mp4"):
        video = self.engine.generate(prompt, num_inference_steps=steps).video[0]
        self.engine.save_video(video, f"{path}/video.mp4")

class OpenSoraT2VideoPipeline():
    def __init__(self, **kwargs):
        config = OpenSoraConfig(**kwargs)
        self.engine = VideoSysEngine(config)
    
    def __call__(self, prompt: str, path: str = "./dataset/temp/video.mp4", **kwargs):
        video = self.engine.generate(prompt, **kwargs).video[0]
        self.engine.save_video(video, f"{path}/video.mp4")

if __name__ == "__main__":
    #     t2v_pipeline = T2VideoPipeline(enable_pab=True, num_gpus=1)
    #     t2v_pipeline("Sunset over the sea", steps=8, path=STORAGE_PATH)
    t2v_pipeline = OpenSoraT2VideoPipeline(num_sampling_steps=30, cfg_scale=7.0, enable_pab=True, num_gpus=1)
    t2v_pipeline("Sunset over the sea", STORAGE_PATH, resolution="480p", aspect_ratio="9:16", num_frames="2s", seed=-1,)
