# 🐣 PlatyPose: EnerGy Guided diffusion for lifting human poses from 2D to 3D

<p align="center">
  <img width="100%" src="assets/logo_wide.png" alt="Platypose Logo">
</p>

This repository is derived from [EGG diffusion](https://github.com/sinzlab/energy-guided-diffusion), hence the name CHICK.

## Abstract
Diffusion models allow us to sample from complex distributions by estimating the reverse diffusion process.
As we show in our previous work [EGG diffusion](https://github.com/sinzlab/energy-guided-diffusion) diffusion models can be guided using an energy function.
In this work we apply this idea to the task of lifting 2D human poses to 3D.
We can 1) learn a powerful pose + motion prior using a diffusion model and 2) define an energy function that guides the diffusion process to generate realistic 3D poses that are consistent with the 2D pose.
We propose to use the re-projection likelihood of the 3D pose as energy function, i.e. the likelihood of the 3D pose projected to 2D given the 2D keypoint heatmap.

## Reproducing the results

#### 1. Download the data
Navigate to the [dataset](https://github.com/sinzlab/chick/dataset) folder and follow the instructions in the `README.md` file.

#### 2. Base docker image
Ensure that you have the base image pulled from the Docker Hub.
You can get the base image by running the following command:
```
docker pull sinzlab/pytorch:v3.9-torch1.13.1-cuda11.7.0-dj0.12.9
```

#### 3. Build the Platypose docker image
First build the docker image by running the following command in the root directory of the repository:
```bash
docker-compose build base
```

#### 4. Run the training
To train the model run the following command:
```bash
docker-compose run -d --name platypose_train_0 -e NVIDIA_VISIBLE_DEVICES=0 train
```
this will spawn a docker container that will start the `main.py` script and will detach the process from the terminal.
the `platypose_train_0` can be any name you want to give to the process. In the example 0 indicates the GPU id this process will use.
To see the output of the training run the following command:
```bash
docker logs platypose_train_0
```

#### 5. Run the evaluation
To evaluate the model run the following command:
```bash
docker-compose run --name platypose_eval_0 -e NVIDIA_VISIBLE_DEVICES=0 eval
```
this will spawn a docker container that will start the `./scrips/eval.py` script.


## Documentation

### Chick sampling
Here is a short snippet on how to load in the model with pretrained weights and how to generate samples from the model. 
```python
from chick.chick import Chick

chick = Chick.from_pretrained("sinzlab/chick/MDM_H36m_1_frame_50_steps:latest")

samples_progressive = chick.sample(
    num_samples=10,
    num_frames=1,
    energy_fn=..., # some energy function
    energy_scale=100,
) # returns a generator of samples for each step of the diffusion process

*_, _sample = samples_progressive # get the final sample

sample = _sample["sample"] # get the 3D pose sample
```

### Experiment Configs
Experiments are configured via YAML files and can be overriden via command line arguments.
The YAML files are located in the `experiments` folder.

An example config file is shown below:
```yaml
experiment:
  num_samples: 50
  energy_scale: 30
  num_frames: 1
  model: "sinzlab/chick/MDM_H36m_1_frame_50_steps:latest"
  seed: 1
  projection: "dummy"
```

You can override the config file by passing the arguments via the command line:
```bash
docker compose run eval --num_samples 100 --energy_scale 50
``` 

### Poetry
This project uses [Poetry](https://python-poetry.org/) to manage the dependencies.
To install the dependencies run the following command:
```bash
poetry install
```
To add a new dependency run the following command:
```bash
poetry add <dependency>
```
It makes sure that the versions are stored in the `pyproject.toml` file and the `poetry.lock` file allowing for reproducible builds.
*Remeber to commit the changes to the `pyproject.toml` and `poetry.lock` files.*
**Remeber to upload the updated `pyproject.toml` and `poetry.lock` files to the server before building the docker image.**