import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from omegaconf import DictConfig

from models.base_model import BaseModel

class Wan21Model(BaseModel):
    def __init__(self, model_config: DictConfig):
        super().__init__(model_config)
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """Loads the pre-trained pipeline."""
        return WanPipeline.from_pretrained(
            self.model_config.model.checkpoint,
            torch_dtype=torch.bfloat16 if self.model_config.model.dtype == 'bf16' else torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.model_config.model.device_map
        )

    def generate(self, prompts: dict, output_path: str) -> None:
        """Generates a video based on the provided prompts."""
        output = self.pipeline(
            prompt=prompts.get("positive", ""),
            negative_prompt=prompts.get("negative", ""),
            num_frames=self.model_config.num_frames,
            width=self.model_config.width,
            height=self.model_config.height,
            guidance_scale=5.0,
        ).frames[0]
        
        export_to_video(output, output_path, fps=16)
        print(f"Video saved to {output_path}")
