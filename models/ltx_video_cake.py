import torch
from diffusers import LTXConditionPipeline
from diffusers.utils import export_to_video, load_image

pipeline = LTXConditionPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16
)

pipeline.load_lora_weights("Lightricks/LTX-Video-Cakeify-LoRA", adapter_name="cakeify")
pipeline.set_adapters("cakeify")

# use "CAKEIFY" to trigger the LoRA
prompt = "CAKEIFY a person using a knife to cut a cake shaped like a Pikachu plushie"
image = load_image("https://huggingface.co/Lightricks/LTX-Video-Cakeify-LoRA/resolve/main/assets/images/pikachu.png")

# pipeline = pipeline.to("cuda")
pipeline.enable_model_cpu_offload() # gpu memory drops from 16.053GB to 309MB on V100
torch.cuda.empty_cache()

video = pipeline(
    prompt=prompt,
    image=image,
    width=512,
    height=512,
    num_frames=80,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)