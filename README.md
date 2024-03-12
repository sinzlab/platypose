# 🐣 Platypose: Calibrated Zero-Shot Multi-Hypothesis 3D Human Motion Estimation

<p align="center">
  <img width="100%" src="assets/logo_wide.png" alt="Platypose Logo">
</p>

> [**Energy Guided Diffusion for Generating Neurally Exciting Images**](https://www.biorxiv.org/content/10.1101/2023.05.18.541176v1), \
> Pawel A. Pierzchlewicz, Caio Oliviera, James Cotton, Fabian H. Sinz
>
> [[arXiv]](https://arxiv.org/abs/2403.06164)

## Reproducing the results

#### 1. Download the data
Navigate to the [dataset](/dataset) folder and follow the instructions in the `README.md` file.

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
docker compose run -d --name platypose_train_0 -e NVIDIA_VISIBLE_DEVICES=0 train --experiment h36m
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
docker compose run --name platypose_eval_0 -e NVIDIA_VISIBLE_DEVICES=0 eval --experiment h36m
```
this will spawn a docker container that will start the `./scrips/eval.py` script.


## Documentation

### Platypose sampling
Here is a short snippet on how to load in the model with pretrained weights and how to generate samples from the model.

```python
from platypose.pipeline import SkeletonPipeline

pipe = SkeletonPipeline.from_pretrained("sinzlab/platypose/MDM_H36m_1_frame_50_steps:latest")

samples_progressive = pipe.sample(
    num_samples=10,
    num_frames=1,
    energy_fn=...,  # some energy function
    energy_scale=100,
)  # returns a generator of samples for each step of the diffusion process

*_, _sample = samples_progressive  # get the final sample

sample = _sample["sample"]  # get the 3D pose sample
```

### Experiment Configs
Experiments are configured via YAML files and can be overriden via command line arguments.
The YAML files are located in the `experiments` folder.

The YAML file overrides the default config file located in `platypose/config.py`.

An example config file is shown below:
```yaml
experiment:
  num_samples: 50
  energy_scale: 30
  num_frames: 1
  model: "sinzlab/platypose/MDM_H36m_1_frame_50_steps:latest"
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