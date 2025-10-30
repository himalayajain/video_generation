import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video


# New, recommended code
import imageio

# The 'frames' variable is a list of NumPy arrays or PIL Images
# Ensure imageio-ffmpeg is installed with `pip install imageio[ffmpeg]`

def custom_export_to_video(frames, filename, fps=16):
# Use imageio.get_writer to create a video file
    with imageio.get_writer(filename, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Video saved to {filename}")

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    # prompt=(
    #     "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
    #     "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
    #     "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
    #     "golden hour, coastal landscape, seaside scenery"
    # ),
    prompt=(
        "masterpiece, best quality, highly detailed, "
        "dynamic action shot of a small, adorable fluffy calico cat, "
        "Sunlit living room, soft morning light, "
        "shallow depth of field, sharp focus on the cat."
    ),
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(42),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")
export_to_video(frames, "output.mp4", fps=16)