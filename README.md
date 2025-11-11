# Video Generation with Multiple Models

This project provides a unified interface for running various text-to-video and image-to-video generation models. It uses Hydra for configuration management, allowing for easy switching between models and adjusting parameters.

## Installation

1.  Create and activate a Conda environment:
    ```bash
    conda create --name video_gen python=3.11
    conda activate video_gen
    ```

2.  Install the required Python packages:
    ```bash
    pip install torch torchvision torchaudio
    pip install diffusers==0.34.0 transformers==4.56.0 accelerate==1.0.0
    pip install imageio imageio-ffmpeg sentencepiece
    pip install hydra-core omegaconf
    pip install peft
    pip install ftfy
    ```
    *Note: Depending on your system and CUDA version, you might need a specific PyTorch installation command. Please refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for details.*

## Project Structure

```
.
├── configs/
│   ├── config.yaml           # Main Hydra config
│   ├── prompts.yaml          # Prompts for video generation
│   └── model/                # Model-specific configurations
│       ├── animatediff.yaml
│       ├── cogvideox.yaml
│       ├── ltx_video_cake.yaml
│       ├── ltx_video.yaml
│       ├── wan21_model.yaml
│       └── wan22_model.yaml
├── models/
│   ├── __init__.py
│   ├── base_model.py         # Base model interface
│   ├── animatediff.py
│   ├── cogvideox.py
│   ├── cogvideox_quantized.py
│   ├── ltx_video_cake.py
│   ├── ltx_video.py
│   ├── wan21_model.py
│   └── wan22_model.py
├── README.md
└── run.py                    # Main script to run the models
```

## Usage

To generate a video, use the `run.py` script. You can specify which model to use via the command line. The script will automatically load the corresponding configuration from the `configs/model` directory.

The general command is:
```bash
python run.py model=<model_name>
```

For example, to run the `animatediff` model:
```bash
python run.py model=animatediff
```

The generated video will be saved as `output.mp4` in the root directory. You can customize prompts by editing `configs/prompts.yaml` and generation parameters (like width, height, number of frames) in `configs/config.yaml`.

## Available Models

This table summarizes the models currently integrated into the project.

| Model Name | Config Name (`model=...`) | Status | Notes |
| :--- | :--- | :--- | :--- |
| AnimateDiff | `animatediff` | ✅ Working | |
| Wan 2.1 | `wan21_model` | ✅ Working | |
| Wan 2.2 | `wan22_model` | ✅ Working | |
| LTX Video Cake | `ltx_video_cake` | ✅ Working | Checkpoint changed to `stabilityai/stable-video-diffusion-img2vid-xt` due to original being private. |
| LTX Video | `ltx_video` | ✅ Working | Checkpoint changed to `stabilityai/stable-video-diffusion-img2vid-xt` due to original being private. |
| CogVideoX | `cogvideox` | ❌ Skipped | Encountered persistent tensor-size mismatch errors. |
| CogVideoX Quantized | `cogvideox_quantized` | ❌ Skipped | Same as `cogvideox`. |

## Adding a New Model from Hugging Face

To add a new model to this framework, follow these steps:

1.  **Create a Model Script:**
    -   Create a new Python file in the `models/` directory (e.g., `models/new_model.py`).
    -   The filename should be the `snake_case` version of the model name.

2.  **Implement the Model Class:**
    -   In your new script, create a class that inherits from `BaseModel`. The class name should be the `PascalCase` version of the model name, ending with `Model` (e.g., `NewModelModel`).
    -   Implement the `__init__`, `_load_pipeline`, and `generate` methods.
    -   The `_load_pipeline` method should load the desired pipeline from `diffusers` using the checkpoint specified in the config.
    -   The `generate` method should call the pipeline with the appropriate parameters and save the output video.

3.  **Create a Configuration File:**
    -   Create a new YAML file in `configs/model/` (e.g., `configs/model/new_model.yaml`).
    -   This file must contain `model_name`, `checkpoint`, `dtype`, and `device_map`.
    -   `model_name` must match the name of your model script (e.g., `new_model`).
    -   `checkpoint` should be the Hugging Face repository ID of the model.

4.  **Update `run.py` (If Necessary):**
    -   The `run.py` script automatically converts `snake_case` model names to `PascalCase` class names.
    -   If your class name does not follow this convention, you will need to add a special case to the `get_model_class` function in `run.py`.
