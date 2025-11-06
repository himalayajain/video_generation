
# Installation
```bash
conda create -n video_gen python=3.11 diffusers transformers accelerate imageio ffmpeg -c conda-forge



conda create --name video_gen python=3.11
conda activate video_gen

python -m pip install diffusers==0.34.0 transformers==4.56.0 accelerate==1.0.0 --index-url https://pypi.org/simple/
# This command installs PyTorch, diffusers, and other necessary libraries
# pip install torch torchvision torchaudio
# pip install diffusers transformers accelerate imageio ffmpeg
# conda install conda-forge::opencv
pip install imageio-ffmpeg
pip install sentencepiece
pip install hydra-core omegaconf
pip install peft
pip install ftfy
# conda install -c conda-forge imageio ffmpeg

# use --index-url https://pypi.org/simple/ if pip is configured to use other python package index
```

