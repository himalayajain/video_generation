import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video, load_image
from omegaconf import DictConfig

from models.base_model import BaseModel

class LtxVideoCakeModel(BaseModel):
    def __init__(self, model_config: DictConfig):
        super().__init__(model_config)
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """Loads the pre-trained pipeline."""
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.model_config.model.checkpoint,
            torch_dtype=torch.bfloat16 if self.model_config.model.dtype == 'bf16' else torch.float16,
            variant="fp16"
        )
        pipe.enable_model_cpu_offload()
        return pipe

    def generate(self, prompts: dict, output_path: str) -> None:
        """Generates a video based on the provided prompts."""
        image = load_image("ref_img.png").resize((1024, 576))
        
        generator = torch.manual_seed(42)
        video_frames = self.pipeline(
            image,
            decode_chunk_size=8,
            generator=generator,
            num_frames=self.model_config.num_frames,
            width=self.model_config.width,
            height=self.model_config.height,
        ).frames[0]
        
        export_to_video(video_frames, output_path, fps=7)
        print(f"Video saved to {output_path}")
