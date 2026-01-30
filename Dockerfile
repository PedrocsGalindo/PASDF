FROM python:3.10-slim

# Dependências de sistema pra Open3D, etc.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# PyTorch CPU oficial
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Demais libs usadas pelo projeto PASDF (sem meshplot)
RUN pip install --no-cache-dir \
    numpy scipy scikit-learn scikit-image matplotlib tqdm yacs pyyaml tabulate termcolor imageio \
    open3d pybullet point-cloud-utils trimesh fvcore iopath plotly tensorboard tensorboard-data-server

CMD ["bash"]