# 🐣 CHICK: Energy guided diffusion for lifting human poses from 2D to 3D

This repository is derived from [EGG diffusion](https://github.com/sinzlab/energy-guided-diffusion), hence the name CHICK.

### Abstract
Diffusion models allow us to sample from complex distributions by estimating the reverse diffusion process.
As we show in our previous work [EGG diffusion](https://github.com/sinzlab/energy-guided-diffusion) diffusion models can be guided using an energy function.
In this work we apply this idea to the task of lifting 2D human poses to 3D.
We can 1) learn a powerful pose + motion prior using a diffusion model and 2) define an energy function that guides the diffusion process to generate realistic 3D poses that are consistent with the 2D pose.
We propose to use the re-projection likelihood of the 3D pose as energy function, i.e. the likelihood of the 3D pose projected to 2D given the 2D keypoint heatmap.