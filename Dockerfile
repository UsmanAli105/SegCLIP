FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# Set up basics
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV PATH=/opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda clean -afy

# Install Python + dependencies in base environment
RUN conda install python=3.8 -y && \
    conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y && \
    pip install mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html && \
    pip install mmsegmentation==0.18.0 && \
    pip install webdataset==0.1.103 && \
    pip install timm==0.4.12 && \
    pip install opencv-python==4.4.0.46 termcolor==1.1.0 diffdist einops omegaconf && \
    pip install nltk ftfy regex tqdm && \
    pip install prefetch_generator && \
    pip install Pillow==8.2.0 && \
    conda clean -afy

# Default to bash
CMD ["/bin/bash"]
