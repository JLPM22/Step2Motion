import json
import os
import torch
import random
import numpy as np
import pymotion.rotations.quat_torch as quat
from argparse import ArgumentParser, Namespace
from dataset import MotionDataset
from collections import defaultdict
from typing import Tuple, Dict
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk

weights = {
    "S1": 91.0,
    "S2": 77.0,
    "S3": 79.0,
    "S4": 65.0,
    "S5": 84.0,
    "S6": 70.0,
    "S7": 88.0,
    "S8": 88.0,
    "S9": 77.0,
}


def find_weight(name: str) -> float:
    for key in weights:
        if key in name:
            return weights[key]
    raise ValueError(f"Unknown weight for {name}")


def main(args: Namespace) -> None:
    # Gather all the files
    files = defaultdict(lambda: ["", "", ""])
    for root, _, filenames in os.walk(args.input_dir):
        dir_name = root.split("\\")[-2]
        for filename in filenames:
            basename = dir_name + "_" + os.path.splitext(filename)[0]
            if filename.endswith(".txt"):  # insole data
                files[basename][0] = os.path.join(root, filename)
            elif filename.endswith(".csv"):  # sync data
                files[basename][1] = os.path.join(root, filename)
            elif filename.endswith(".bvh"):  # pose data
                files[basename][2] = os.path.join(root, filename)
            elif not filename.endswith(".fbx"):  # skip fbx files
                print(f"Unknown file: {filename}")

    # Custom split: all S4 files for test, rest split 80% train, 20% val
    keys = list(files.keys())
    test_files = [k for k in keys if "S4" in k or os.path.normpath(files[k][0]).split(os.sep)[-3] == "S4"]
    other_files = [k for k in keys if k not in test_files]
    random.shuffle(other_files)
    num_train = int(len(other_files) * 0.99)
    train_files = other_files[:num_train]
    val_files = other_files[num_train:]
    print(f"Number of files for - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    print("Train files: ", train_files)
    print("Val files: ", val_files)
    print("Test files: ", test_files)
    with open(os.path.join(args.input_dir, f"{args.name}_split.json"), "w") as f:
        json.dump({"train": train_files, "val": val_files, "test": test_files}, f)

    # Process and sync data
    files_data = {}
    for file, [insole_file, sync_file, pose_file] in files.items():
        if insole_file == "" or sync_file == "" or pose_file == "":
            raise ValueError(f"Missing file for {file}")

        print(f"Processing {file} -----------------")
        insole_data = process_insole_file(insole_file, find_weight(file))
        sync_data = process_sync_file(sync_file)
        pose_data, quats, parents, global_pos, global_rot, offsets = process_pose_file(pose_file)
        print(f"Processed {file} - Insole: {insole_data.shape}, Pose: {pose_data.shape}, Sync: {sync_data}")

        insole_data, pose_data, quats, global_pos, global_rot = sync_insole_to_pose(
            insole_data, pose_data, quats, global_pos, global_rot, sync_data, args
        )
        print(f"Synced {file} - Insole: {insole_data.shape}, Pose: {pose_data.shape}")

        # insole_data = remove_drift_insole_data(insole_data)

        displacements = torch.cat(
            [
                torch.tensor([[0.0, 0.0, 0.0]], device=global_pos.device),
                global_pos[1:] - global_pos[:-1],
            ],
            dim=0,
        )

        files_data[file] = (insole_data, pose_data, quats, displacements, parents, offsets, global_rot)

    # Create the datasets
    def create_dataset(files: list, name: str) -> MotionDataset:  # type: ignore
        data = []
        for file in files:
            data.append(files_data[file])
        insole = torch.cat([d[0] for d in data], dim=0)
        pose = torch.cat([d[1] for d in data], dim=0)
        quats = torch.cat([d[2] for d in data], dim=0)
        displacements = torch.cat([d[3] for d in data], dim=0)
        offsets = torch.cat([d[5][None, ...] for d in data], dim=0)
        global_rots = torch.cat([d[6] for d in data], dim=0)
        clips = [0]
        for d in data[:-1]:
            clips.append(clips[-1] + d[1].shape[0])
        dataset = MotionDataset(
            clips=clips,
            poses=pose,
            quats=quats,
            displacements=displacements,
            global_rots=global_rots,
            insole=insole,
            parents=data[0][4],
            offsets=offsets,
            is_acceleration_world=args.acc_world,
            target_sample_rate=args.target_hz,
            foot_indices=[7, 8, 3, 4],
        )
        print(f"{name} - Insole: {insole.shape}, Pose: {pose.shape}")
        torch.save(dataset, os.path.join(args.input_dir, f"{args.name}_{name}.pt"))
        return dataset

    if len(train_files) > 0:
        create_dataset(train_files, "train")
    if len(val_files) > 0:
        create_dataset(val_files, "val")
    if len(test_files) > 0:
        create_dataset(test_files, "test")


def process_insole_file(path: str, weight: float) -> torch.Tensor:
    with open(path, "r") as f:
        lines = f.readlines()
        insole_data = []
        for line in lines:
            line = line.strip()
            if line[0] == "#":
                continue
            # missing values are replaced with 0.0
            line = [0.0 if v == "" else float(v) for v in line.split("\t")]
            if len(line) < 51:
                line.extend([0.0] * (51 - len(line)))
            insole_data.append(line)
        insole_data = torch.tensor(insole_data)
        # # left pressure
        # insole_data[:, 1:17] /= weight
        # # left total force
        # insole_data[:, 23] /= weight
        # # right pressure
        # insole_data[:, 26:42] /= weight
        # # right total force
        # insole_data[:, 48] /= weight
    return insole_data


def process_sync_file(path: str) -> Dict[str, Tuple[int, int]]:
    with open(path, "r") as f:
        lines = f.readlines()
        sync_data = {}
        for line in lines[1:]:
            line = line.strip().split(",")
            sync_data[line[0]] = (int(line[1]), int(line[2]))
    return sync_data


def process_pose_file(
    path: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    remove_joints = [11]
    """
        0, # hips
        5, 6, 7, 8, # right leg -> left leg
        1, 2, 3, 4, # left leg -> right leg
        9, 10, 11, # spine
        14, 15, 16, 17, # left arm -> shift 2 (neck, head)
        18, 19, 20, 21, # right arm -> shift 2 (neck, head)
        12, 13 # neck, head
    """
    reorder_joints = [
        0,
        5,
        6,
        7,
        8,
        1,
        2,
        3,
        4,
        9,
        10,
        11,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        12,
        13,
    ]
    bvh = BVH()
    bvh.load(path)
    bvh.remove_joints(remove_joints)
    bvh.set_order_joints(reorder_joints)
    local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()
    root_joint_idx = 0
    global_positions = local_positions[:, root_joint_idx, :]
    global_rotations = local_rotations[:, root_joint_idx, :].copy()
    pos, rotmats = fk(local_rotations, np.zeros((3)), offsets, parents)
    pos = torch.from_numpy(pos[:, 1:, :]).flatten(-2, -1)  # remove root joint (always 0, 0, 0 in root space)
    quats = quat.from_matrix(torch.from_numpy(rotmats))
    global_positions = torch.from_numpy(global_positions)
    global_rotations = torch.from_numpy(global_rotations)
    return (
        pos,
        quats,
        torch.from_numpy(parents),
        global_positions,
        global_rotations,
        torch.from_numpy(offsets),
    )


def sync_insole_to_pose(
    insole_data: torch.Tensor,
    pose_data: torch.Tensor,
    quats: torch.Tensor,
    global_pos: torch.Tensor,
    global_rot: torch.Tensor,
    sync_data: Dict[str, Tuple[int, int]],
    args: Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left_insole_data = insole_data[:, 1:26]
    right_insole_data = insole_data[:, 26:]
    start_left_insole, end_left_insole = sync_data["left_insole"]
    start_right_insole, end_right_insole = sync_data["right_insole"]
    start_pose, end_pose = sync_data["mocap"]

    target_nframes = int(round((end_pose - start_pose) * (args.target_hz / args.pose_hz)))
    # Trim start and end
    left_insole_data = left_insole_data[start_left_insole:end_left_insole]
    right_insole_data = right_insole_data[start_right_insole:end_right_insole]
    pose_data = pose_data[start_pose:end_pose]
    quats = quats[start_pose:end_pose]
    global_pos = global_pos[start_pose:end_pose]
    global_rot = global_rot[start_pose:end_pose]

    # Interpolate data
    left_insole_data = interpolate_data(left_insole_data, target_nframes)
    right_insole_data = interpolate_data(right_insole_data, target_nframes)
    pose_data = interpolate_data(pose_data, target_nframes)
    quats = quats.reshape(quats.shape[0], -1)
    quats = interpolate_data(quats, target_nframes, mode="nearest")
    quats = quats.reshape(quats.shape[0], -1, 4)
    global_pos = interpolate_data(global_pos, target_nframes)
    global_rot = interpolate_data(global_rot, target_nframes, mode="nearest")

    return torch.cat([left_insole_data, right_insole_data], dim=1), pose_data, quats, global_pos, global_rot


def interpolate_data(data: torch.Tensor, target_nframes: int, mode: str = "linear") -> torch.Tensor:
    nframes = len(data)
    if nframes == target_nframes:
        return data
    else:
        new_data = torch.zeros(target_nframes, data.shape[1])
        for i in range(data.shape[1]):
            new_data[:, i] = torch.nn.functional.interpolate(
                data[:, i].unsqueeze(0).unsqueeze(0), target_nframes, mode=mode
            ).squeeze()
        return new_data


def remove_drift_insole_data(data: torch.Tensor) -> torch.Tensor:
    l_acceleration_idx = (16, 19)
    r_acceleration_idx = (41, 44)

    earth_acc = 9.81
    zero_threshold = 1.5  # threshold in m/s/s
    zero_threshold /= earth_acc
    gravity_dir = torch.tensor([0.0, 0.0, 1.0])

    def remove_drift(acc_idx):
        acceleration = data[:, slice(*acc_idx)].clone()
        imu_isnotmoving = torch.empty((len(acceleration),)).int()
        for index in range(len(imu_isnotmoving)):
            # detect if not moving and feet on the ground
            imu_isnotmoving[index] = (
                torch.sqrt(acceleration[index].dot(acceleration[index])) - 1 < zero_threshold
            ) and (acceleration[index] / torch.linalg.norm(acceleration[index])).dot(gravity_dir) > 0.8

        drift = acceleration[imu_isnotmoving.bool()].clone()
        drift[:, 2] -= 1  # remove gravity
        acceleration -= drift.mean(dim=0)
        data[:, slice(*acc_idx)] = acceleration

    remove_drift(l_acceleration_idx)
    remove_drift(r_acceleration_idx)

    return data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name", type=str, help="Name of the dataset")
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--pose_hz", type=float, default=240.0)
    parser.add_argument("--target_hz", type=float, default=30.0)
    parser.add_argument("--acc_world", action="store_true", default=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"Processing {args.name} dataset -----------------")

    main(args)
