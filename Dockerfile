FROM nvidia/cuda:11.1.1-runtime-ubuntu20.04

# Install only what you actually need at runtime
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to avoid old wheel issues
RUN pip3 install --upgrade pip

# Install dependencies (PyTorch + libs)
RUN pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install mmcv-full==1.3.14 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html && \
    pip3 install \
      mmsegmentation==0.18.0 \
      webdataset==0.1.103 \
      timm==0.4.12 \
      opencv-python==4.4.0.46 \
      termcolor==1.1.0 \
      diffdist einops omegaconf \
      nltk ftfy regex tqdm prefetch_generator \
      Pillow==8.2.0 && \
    # clean up pip caches
    pip cache purge && \
    rm -rf ~/.cache/pip

# Default command
CMD ["/bin/bash"]
