import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from accelerate import Accelerator

prompt = "Abstract art to reflect a scared person's aura"

# Initialize the Accelerator and set the device to GPU explicitly
accelerator = Accelerator()

# Set device for CUDA explicitly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pipeline and move it to the GPU
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16
).to(device)

# Prepare the pipeline for Accelerate
pipe = accelerator.prepare(pipe)

# Enable only VAE optimizations for efficiency, while keeping the model on GPU
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Generate the video, making sure that input is on the GPU
with accelerator.autocast():
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device=device).manual_seed(42),
    ).frames[0]

# Export the video (only on the main process)
if accelerator.is_main_process:
    export_to_video(video, "output.mp4", fps=8)
