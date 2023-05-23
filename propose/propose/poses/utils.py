import functools

import yaml


def load_data_ids(path: str) -> list:
    """
    Loads the data_ids of a pose.
    :param path: Path to the yaml file.
    :return: A list containing the data_ids of the pose.
    """
    with open(path, "r") as f:
        metadata = yaml.safe_load(f)

    n_joints = sum([len(metadata[group].keys()) for group in metadata])

    data_ids = [None] * n_joints
    for group in metadata:
        group_data = metadata[group]
        for joint in group_data:
            joint_data = group_data[joint]
            data_ids[joint_data["id"]] = joint_data["data_id"]

    return data_ids


@functools.lru_cache()
def yaml_pose_loader(path: str) -> tuple[list, list, dict]:
    """
    Loads a yaml file containing the metadata of a pose.
    :param path: Path to the yaml file.
    :return: A tuple containing the metadata of the pose. Named edges and grouped edges.
    """
    with open(path, "r") as f:
        metadata = yaml.safe_load(f)

    n_joints = sum([len(metadata[group].keys()) for group in metadata])

    edges = []
    group_edges = {}
    marker_names = [""] * n_joints

    for group in metadata:
        group_data = metadata[group]
        group_edges[group] = []
        for joint in group_data:
            joint_data = group_data[joint]

            marker_names[joint_data["id"]] = joint

            if joint_data["parent_id"] >= 0:
                edge = (joint_data["parent_id"], joint_data["id"])

                edges.append(edge)
                group_edges[group].append(edge)

    named_edges = [(marker_names[src], marker_names[dst]) for src, dst in edges]
    named_group_edges = {
        group: [
            (marker_names[src], marker_names[dst]) for src, dst in group_edges[group]
        ]
        for group in group_edges.keys()
    }

    return marker_names, named_edges, named_group_edges
