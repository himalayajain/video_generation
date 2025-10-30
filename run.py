import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import os
import importlib

def get_model_class(model_name: str):
    """Dynamically import and return the model class."""
    module_path = f"models.{model_name}"
    # Convert snake_case to PascalCase for the class name
    class_name = "".join(word.capitalize() for word in model_name.split('_'))
    if not class_name.endswith("Model"):
        class_name += "Model"
    
    # Special case for single-word model names
    if '_' not in model_name:
        if model_name == "animatediff":
            class_name = "AnimateDiffModel"
        elif model_name == "cogvideox":
            class_name = "CogVideoXModel"
        elif model_name == "cogvideox_quantized":
            class_name = "CogVideoXQuantizedModel"
        elif model_name == "ltx_video":
            class_name = "LtxVideoModel"
        elif model_name == "ltx_video_cake":
            class_name = "LtxVideoCakeModel"
        elif model_name == "wan22_model":
            class_name = "Wan22Model"
        else:
            class_name = model_name.capitalize() + "Model"
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to instantiate a model and run inference via its API.
    """
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    model_name = cfg.model.model_name
    
    # Get the model class
    ModelClass = get_model_class(model_name)
    
    # Instantiate the model
    model = ModelClass(cfg)

    # Load prompts from the YAML file
    with open("configs/prompts.yaml", 'r') as f:
        prompts = yaml.safe_load(f)

    # Define the output path
    output_video_path = os.path.join(os.getcwd(), "output.mp4")
    
    # Generate the video
    model.generate(prompts, output_video_path)

    print("Video generation complete.")

if __name__ == "__main__":
    main()
