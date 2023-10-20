# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import pickle
import sys

import numpy as np
import smplx
import torch

from trimesh import Trimesh

sys.path.append('/src')

"""
- load a sequence
- create a npy2obj class
- iterate over frames and save the converted obj
"""

def get_trimesh(vertices, faces, frame):
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    return Trimesh(vertices[frame, ...], faces, vertex_colors=vertex_colors)

def save_obj(save_path, frame, vertices, faces):
    mesh = get_trimesh(vertices, faces, frame)
    with open(save_path, "w") as fw:
        mesh.export(fw, "obj")
        

if __name__ == "__main__":
    with open("./dataset/h3dpw.pkl", "rb") as f:
        data = pickle.load(f)
        sequence = np.array(data["S9"])
        n_frames = sequence.shape[0]

    model = smplx.create(
        model_path="./body_models",
        model_type="smpl",
        gender="neutral",
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext="pkl"
    )
    print(model)

    global_orient = torch.Tensor(sequence[:, :3])
    body_pose = torch.Tensor(sequence[:, 3:])

    output = model(
        global_orient=global_orient,
        body_pose=body_pose,
        return_verts=True
    )

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    for frame in range(n_frames):
        save_obj(f"examples/first_{frame}.obj", frame, vertices, model.faces)
