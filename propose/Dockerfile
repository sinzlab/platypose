ARG BASE_IMAGE=sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7

ARG TORCH_VERSION="torch-1.9.0"
ARG CUDA_VERSION="cu111"

# Perform multistage build to pull private repo without leaving behind
# private information (e.g. SSH key, Git token)
FROM ${BASE_IMAGE} as base
ARG DEV_SOURCE=sinzlab
ARG GITHUB_USER
ARG GITHUB_TOKEN

WORKDIR /src
# Use git credential-store to specify username and pass to use for pulling repo
RUN git config --global credential.helper store &&\
    echo https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com >> ~/.git-credentials

FROM ${BASE_IMAGE}

ADD . /src/propose

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get -yy update
RUN apt-get install -y ffmpeg

RUN pip install -e /src/propose

RUN python -m pip install --no-cache-dir nflows\
    imageio\
    tqdm\
    torch-geometric\
    ffmpeg-python\
    scikit-image\
    cdflib\
    imageio-ffmpeg\
    brax\
    wandb\
    neuralpredictors\
    yacs

RUN pip install --upgrade pillow

RUN pip install git+https://github.com/sinzlab/neuralpredictors.git
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
RUN pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

WORKDIR /
