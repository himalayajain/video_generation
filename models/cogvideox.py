import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel
# torchao is no longer needed for this approach
# from torchao.quantization import quantize_, int8_weight_only

# Using float16 is best for V100 GPUs
dtype = torch.bfloat16

# --- Load models WITHOUT quantization ---
print("Loading models (without quantization)...")
text_encoder = T5EncoderModel.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="text_encoder", torch_dtype=dtype)
transformer = CogVideoXTransformer3DModel.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="transformer", torch_dtype=dtype)
vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=dtype)
print("Models loaded.")

# --- Create the pipeline ---
# Notice we are now passing the full-precision models
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=dtype,
)

# --- APPLY MEMORY-SAVING OPTIMIZATIONS ---
# This will now work because the models use standard PyTorch tensors
pipe.enable_sequential_cpu_offload()

# VAE tiling is still a good idea
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

print("Running inference...")
prompt = "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
image = load_image("input_smaller.jpg") 

# CRITICAL: Keep the number of frames reasonable. Offloading helps, but isn't magic.
# Start with 24 and see if it works.
num_frames_to_generate = 24 

torch.cuda.empty_cache()

# --- Run Inference ---
video = pipe(
    prompt=prompt,
    image=image,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=num_frames_to_generate,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)
print(f"Successfully generated 'output.mp4' with {num_frames_to_generate} frames.")