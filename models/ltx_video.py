import torch
from diffusers import LTXPipeline, AutoModel
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video

# fp8 layerwise weight-casting
transformer = AutoModel.from_pretrained(
    "Lightricks/LTX-Video",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
)

pipeline = LTXPipeline.from_pretrained("Lightricks/LTX-Video", transformer=transformer, torch_dtype=torch.bfloat16)

# group-offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True)
apply_group_offloading(pipeline.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2)
apply_group_offloading(pipeline.vae, onload_device=onload_device, offload_type="leaf_level")

prompt = """
A woman with long brown hair and light skin smiles at another woman with long blonde hair.
The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek.
The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and 
natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage
"""

prompt = """A boy playing with a small dog on the bed. The room is well-lit with natural light coming through a window. It is a close-up shot, appears to be real-life footage"""
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=65,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
torch.cuda.empty_cache()