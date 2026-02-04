FROM mambaorg/micromamba:1.5.8

WORKDIR /workspace

USER root
RUN apt-get update && apt-get install -y \
    libx11-6 libxext6 libxrender1 libsm6 \
    libgl1 libglib2.0-0 libgomp1 \
    git build-essential cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

COPY environment_linux.yaml /tmp/environment_linux.yaml

# Cria env base
RUN micromamba create -y -n deepsdf -f /tmp/environment_linux.yaml && \
    micromamba clean -a -y

# ✅ GARANTIR que torch existe no env (CPU-only)
# (mesmo se o YAML já tiver, isso força e evita o erro do build do pytorch3d)
RUN micromamba run -n deepsdf micromamba install -y -n deepsdf -c pytorch \
    pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 && \
    micromamba clean -a -y

# pip + typing_extensions atualizado
RUN micromamba run -n deepsdf python -m pip install -U pip "typing_extensions>=4.10"

# (opcional) sanity check: confirmar torch importável
RUN micromamba run -n deepsdf python -c "import torch; print('torch ok', torch.__version__)"

# ✅ PyTorch3D CPU: compila do source (agora torch já existe)
RUN micromamba run -n deepsdf python -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4"

ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH=/opt/conda/envs/deepsdf/bin:$PATH
ENV PYTHONPATH=/workspace
SHELL ["bash", "-lc"]
CMD ["bash"]
