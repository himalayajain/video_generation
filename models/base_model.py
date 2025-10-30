from abc import ABC, abstractmethod
from omegaconf import DictConfig

class BaseModel(ABC):
    """
    Abstract base class for all models.
    """
    def __init__(self, model_config: DictConfig):
        self.model_config = model_config

    @abstractmethod
    def generate(self, prompts: dict, output_path: str) -> None:
        """
        Generate a video based on the given prompts and save it to the output path.

        Args:
            prompts (dict): A dictionary containing 'positive' and 'negative' prompts.
            output_path (str): The path to save the generated video.
        """
        pass
