import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_video
from omegaconf import DictConfig

from models.base_model import BaseModel

class AnimateDiffModel(BaseModel):
    def __init__(self, model_config: DictConfig):
        super().__init__(model_config)
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """Loads the pre-trained pipeline."""
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
        pipe = AnimateDiffPipeline.from_pretrained(
            self.model_config.model.checkpoint,
            motion_adapter=adapter,
            torch_dtype=torch.float16
        )
        scheduler = DDIMScheduler.from_pretrained(
            self.model_config.model.checkpoint,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        pipe.scheduler = scheduler
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
        return pipe

    def generate(self, prompts: dict, output_path: str) -> None:
        """Generates a video based on the provided prompts."""
        output = self.pipeline(
            prompt=prompts.get("positive", ""),
            negative_prompt=prompts.get("negative", ""),
            num_frames=self.model_config.num_frames,
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=torch.Generator("cpu").manual_seed(42),
        ).frames[0]
        
        export_to_video(output, output_path, fps=16)
        print(f"Video saved to {output_path}")
