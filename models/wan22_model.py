import torch
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from omegaconf import DictConfig

from models.base_model import BaseModel

class Wan22Model(BaseModel):
    def __init__(self, model_config: DictConfig):
        super().__init__(model_config)
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """Loads the pre-trained pipeline."""
        dtype = torch.bfloat16 if self.model_config.model.dtype == 'bf16' else torch.float16
        
        vae = AutoencoderKLWan.from_pretrained(self.model_config.model.checkpoint, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(self.model_config.model.checkpoint, vae=vae, torch_dtype=dtype)
        pipe.to(self.model_config.model.device_map)
        return pipe

    def generate(self, prompts: dict, output_path: str) -> None:
        """Generates a video based on the provided prompts."""
        output = self.pipeline(
            prompt=prompts.get("positive", ""),
            negative_prompt=prompts.get("negative", ""),
            height=self.model_config.height,
            width=self.model_config.width,
            num_frames=self.model_config.num_frames,
            guidance_scale=5.0,
            num_inference_steps=50,
        ).frames[0]
        
        export_to_video(output, output_path, fps=24)
        print(f"Video saved to {output_path}")
