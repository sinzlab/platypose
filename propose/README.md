# PROPOSE

**PRO**babilistic **POSE** estimation

[![Test](https://github.com/PPierzc/propose/workflows/Test/badge.svg)](https://github.com/PPierzc/propose/actions/workflows/test.yml)
[![Black](https://github.com/PPierzc/propose/workflows/Black/badge.svg)](https://github.com/PPierzc/propose/actions/workflows/black.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/PPierzc/propose/branch/main/graph/badge.svg?token=PYI1Z06426)](https://codecov.io/gh/PPierzc/propose)

## Getting started
Install the package from source:
```shell
pip install git+https://github.com/sinzlab/propose.git
```
### Loading a pretrained cGNF
We provide the pretrained model which you can load with the following code snippet.
```python
from propose.models.flows import CondGraphFlow

flow = CondGraphFlow.from_pretrained('ppierzc/cgnf/cgnf_human36m:best')
```

#### HRNet Loading
You can also load a pretrained HRNet model.
```python
from propose.models.detectors import HRNet

hrnet = HRNet.from_pretrained('ppierzc/cgnf/hrnet:v0')
```
This will load the HRNet model provided in the [repo](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).
The model loaded here is the `pose_hrnet_w32_256x256` trained on the MPII dataset.

### Requirements
#### Requirements for the package
The requirements for the package can be found in the [requirements.txt](/requirements.txt).

#### Docker
Alternatively, you can use [Docker](https://www.docker.com/) to run the package.
This project requires that you have the following installed:
- `docker`
- `docker-compose`

Ensure that you have the base image pulled from the Docker Hub.
You can get the base image by running the following command:
```
docker pull sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7
```

### Running notebooks
1. Clone the repository.
2. Navigate to the project directory. 
3. Run```docker-compose build base```
4. Run```docker-compose run -d -p 10101:8888 notebook_server```
5. You can now open JupyterLab in your browser at [`http://localhost:10101`](http://localhost:10101).

#### Available Models
| Model Name | description                                                                                                                                             | Artifact path                                 | Import Code                      |
| --- |---------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|----------------------------------|
| cGNF Human 3.6m | Model trained on the Human 3.6M dataset with MPII input keypoints.                                                                                      | ```ppierzc/propose_human36m/mpii-prod:best``` | ```from propose.models.flows import CondGraphFlow``` |
 | HRNet | Instance of the [official](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) HRNet model trained on the MPII dataset with w32 and 256x256 | ```ppierzc/cgnf/hrnet:v0```                   | ```from propose.models.detectors import HRNet``` |

### Run Tests
To run the tests, from the root directory call:
```
docker-compose run pytest tests
```
 
*Note: This will create a separate image from the base service.*

## Data
### Rat7m
You can download the Rat 7M dataset from [here](https://figshare.com/collections/Rat_7M/5295370).

### Human3.6M dataset
Due to license restrictions, the dataset is not included in the repository.
You can download it from the official [website](http://vision.imar.ro/human3.6m).
