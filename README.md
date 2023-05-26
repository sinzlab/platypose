# 🐣 CHICK: Energy guided diffusion for lifting human poses from 2D to 3D

<p align="center">
  <img width="100%" src="assets/logo_wide.png" alt="Hive Logo">
</p>

This repository is derived from [EGG diffusion](https://github.com/sinzlab/energy-guided-diffusion), hence the name CHICK.

### Abstract
Diffusion models allow us to sample from complex distributions by estimating the reverse diffusion process.
As we show in our previous work [EGG diffusion](https://github.com/sinzlab/energy-guided-diffusion) diffusion models can be guided using an energy function.
In this work we apply this idea to the task of lifting 2D human poses to 3D.
We can 1) learn a powerful pose + motion prior using a diffusion model and 2) define an energy function that guides the diffusion process to generate realistic 3D poses that are consistent with the 2D pose.
We propose to use the re-projection likelihood of the 3D pose as energy function, i.e. the likelihood of the 3D pose projected to 2D given the 2D keypoint heatmap.

### How to use

#### 1. Download the data
Navigate to the [dataset](https://github.com/sinzlab/chick/dataset) folder and follow the instructions in the `README.md` file.

#### 2. Base docker image
Ensure that you have the base image pulled from the Docker Hub.
You can get the base image by running the following command:
```
docker pull sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7
```

#### 3. Build the CHICK docker image
First build the docker image by running the following command in the root directory of the repository:
```bash
docker-compose build base
```

#### 4. Run the training
To train the model run the following command:
```bash
docker-compose run -d --name chick_train_0 -e NVIDIA_VISIBLE_DEVICES=0 train
```
this will spawn a docker container that will start the `main.py` script and will detach the process from the terminal.
the `chick_train_0` can be any name you want to give to the process. In the example 0 indicates the GPU id this process will use.
To see the output of the training run the following command:
```bash
docker logs chick_train_0
```
