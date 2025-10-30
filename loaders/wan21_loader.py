import torch.nn as nn
from models.wan21_model import Wan21Model

class Wan21ModelLoader:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = Wan21Model().to(self.device)

    def download_weights(self, url):
        """
        Fake download function for local dev.
        Replace with real downloader when online access available.
        """
        print(f"Simulating download of weights from: {url}")
        return "local_path_to_weights.pth"

    def load_weights(self, weights_path):
        """
        Load weights into model.
        In first dev step, loads random weights for demonstration.
        """
        print(f"Loading weights from: {weights_path}")
        # Simulate weights loading here with random initialization
        # Replace with torch.load(weights_path, map_location=self.device) when available
        self.model.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def get_model(self):
        return self.model
