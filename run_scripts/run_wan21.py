import torch
from loaders.wan21_loader import Wan21ModelLoader


if __name__ == "__main__":
    loader = Wan21ModelLoader(device='cpu')
    weights_path = loader.download_weights("https://fake-url-to-wan21-weights")
    loader.load_weights(weights_path)
    model = loader.get_model()

    dummy_input = torch.randn(2, 768)  # Simulated input shape
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")