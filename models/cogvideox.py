import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel
from omegaconf import DictConfig

from models.base_model import BaseModel

class CogVideoXModel(BaseModel):
    def __init__(self, model_config: DictConfig):
        super().__init__(model_config)
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """Loads the pre-trained pipeline."""
        dtype = torch.bfloat16 if self.model_config.model.dtype == 'bf16' else torch.float16
        
        text_encoder = T5EncoderModel.from_pretrained(self.model_config.model.checkpoint, subfolder="text_encoder", torch_dtype=dtype)
        transformer = CogVideoXTransformer3DModel.from_pretrained(self.model_config.model.checkpoint, subfolder="transformer", torch_dtype=dtype)
        vae = AutoencoderKLCogVideoX.from_pretrained(self.model_config.model.checkpoint, subfolder="vae", torch_dtype=dtype)

        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            self.model_config.model.checkpoint,
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            torch_dtype=dtype,
        )
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        return pipe

    def generate(self, prompts: dict, output_path: str) -> None:
        """Generates a video based on the provided prompts."""
        # This model requires a specific image, we'll load it here.
        # In a more advanced implementation, this could be specified in the config.
        image = load_image("input_smaller.jpg").resize((self.model_config.width, self.model_config.height))

        output = self.pipeline(
            prompt=prompts.get("positive", ""),
            image=image,
            height=self.model_config.height,
            width=self.model_config.width,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=self.model_config.num_frames,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]
        
        export_to_video(output, output_path, fps=8)
        print(f"Video saved to {output_path}")
