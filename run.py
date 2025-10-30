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
    model = ModelClass(cfg.model)

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
