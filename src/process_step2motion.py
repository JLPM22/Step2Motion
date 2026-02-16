import json
import os
import random
import numpy as np
import torch
import pymotion.rotations.quat_torch as quat
from collections import defaultdict
from argparse import ArgumentParser, Namespace
from typing import Tuple
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk
from datetime import datetime
from dataset import MotionDataset


def main(args: Namespace) -> None:
    # Gather all the files
    files = defaultdict(lambda: ["", "", ""])
    for root, _, filenames in os.walk(args.input_dir):
        for filename in filenames:
            basename = os.path.splitext(filename)[0]
            if filename.endswith(".txt"):  # insole data
                files[basename][0] = os.path.join(root, filename)
            elif filename.endswith(".json"):  # sync data
                files[basename][1] = os.path.join(root, filename)
            elif filename.endswith(".bvh"):  # pose data
                files[basename][2] = os.path.join(root, filename)
            else:
                print(f"Unknown file: {filename}")

    # Remove files with missing data
    files = {k: v for k, v in files.items() if "" not in v}

    # Shuffle the files
    keys = list(files.keys())
    random.shuffle(keys)

    # Split the data in train, val, test
    num_files = len(files)
    if num_files == 1:
        print("Only one file found, skipping split")
        num_train = 1
        num_val = 1
        num_test = 1
        train_files = [keys[0]]
        val_files = [keys[0]]
        test_files = [keys[0]]
    else:
        num_train = num_files - 1
        num_val = 1
        num_test = 1
        train_files = keys[:num_train]
        val_files = keys[num_train:]
        test_files = keys[num_train:]
    print(f"Number of files for - Train: {num_train}, Val: {num_val}, Test: {num_test}")
    print("Train files: ", train_files)
    print("Val files: ", val_files)
    print("Test files: ", test_files)
    # with open(os.path.join(args.input_dir, f"{args.name}_split.json"), "w") as f:
    #     json.dump({"train": train_files, "val": val_files, "test": test_files}, f)

    # Process the data
    files_data = {}
    for file, [insole_file, sync_file, pose_file] in files.items():
        insole_data, insole_start_time = process_insole_file(insole_file)
        sync_data = process_sync_file(sync_file)
        pose_data, quats, parents, global_pos, global_rot, offsets = process_pose_file(pose_file, args.xsens)
        print(f"Processed {file} - Insole: {insole_data.shape}, Pose: {pose_data.shape}")

        trimmed_insole_data = sync_insole_to_pose(insole_data, insole_start_time, sync_data, args)
        # Calculate the number of frames to trim from the end
        delta_time_pose = 1.0 / args.pose_hz  # in seconds
        delta_time_insole = 1.0 / args.insole_hz  # in seconds
        pose_duration = pose_data.shape[0] * delta_time_pose
        insole_duration = len(trimmed_insole_data) * delta_time_insole
        if insole_duration >= pose_duration:
            frames_to_trim_end = int((insole_duration - pose_duration) / delta_time_insole)
            # Trim the insole_data tensor
            trimmed_insole_data = trimmed_insole_data[: len(trimmed_insole_data) - frames_to_trim_end, ...]
        else:
            frames_to_trim_end = int((pose_duration - insole_duration) / delta_time_pose)
            # Trim the pose_data tensor
            pose_data = pose_data[: pose_data.shape[0] - frames_to_trim_end, ...]
            global_pos = global_pos[: global_pos.shape[0] - frames_to_trim_end, ...]
            quats = quats[: quats.shape[0] - frames_to_trim_end, ...]
            global_rot = global_rot[: global_rot.shape[0] - frames_to_trim_end, ...]

        # Interpolate the insole data to match the pose data
        target_nframes = int(round(pose_data.shape[0] * (args.target_hz / args.pose_hz)))
        insole_data = interpolate_data(trimmed_insole_data, target_nframes)
        pose_data = interpolate_data(pose_data, target_nframes)
        quats = quats.reshape(quats.shape[0], -1)
        quats = interpolate_data(quats, target_nframes, mode="nearest")
        quats = quats.reshape(quats.shape[0], -1, 4)
        global_pos = interpolate_data(global_pos, target_nframes)
        global_rot = interpolate_data(global_rot, target_nframes, mode="nearest")

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

    # Only for dancing dataset ---
    # start = [0, 8400, 11280, 21600]
    # end = [8400, 11280, 21600, 24010]
    # data = files_data["0"]
    # for i, (s, e) in enumerate(zip(start, end)):
    #     files_data[str(i + 1)] = (
    #         data[0][s:e],
    #         data[1][s:e],
    #         data[2][s:e],
    #         data[3][s:e],
    #         data[4],
    #         data[5],
    #         data[6][s:e],
    #     )
    # del files_data["0"]
    # train_files = ["1", "3"]
    # val_files = ["2", "4"]
    # test_files = ["2", "4"]
    # ----------------------------

    if len(train_files) > 0:
        create_dataset(train_files, "train")
    if len(val_files) > 0:
        create_dataset(val_files, "val")
    if len(test_files) > 0:
        create_dataset(test_files, "test")


def process_insole_file(path: str) -> Tuple[torch.Tensor, str]:
    start_time = None
    with open(path, "r") as f:
        lines = f.readlines()
        insole_data = []
        for line in lines:
            line = line.strip()
            if line[0] == "#":
                if "Start time" in line:
                    start_time = line.split("Start time:")[1].strip()
                continue
            # missing values are replaced with 0.0
            line = [0.0 if v == "" else float(v) for v in line.split("\t")]
            if len(line) < 51:
                line.extend([0.0] * (51 - len(line)))
            insole_data.append(line)
        insole_data = torch.tensor(insole_data)
    assert start_time is not None, "Start time not found in insole file"
    return insole_data[..., 1:], start_time


def process_sync_file(path: str) -> str:
    with open(path, "r") as f:
        sync_data = json.load(f)
    iso_time = sync_data["insoles"]["start_calibration_ISO"]
    return iso_time


def process_pose_file(
    path: str, is_xsens: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not is_xsens:
        remove_joints = [
            3,
            5,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
        ]
        reorder_joints = [
            0,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]
    bvh = BVH()
    bvh.load(path)
    if not is_xsens:
        bvh.remove_joints(remove_joints)
        bvh.set_scale(0.001)
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
    insole_data: torch.Tensor, insole_start_time: str, sync_data: str, args: Namespace
) -> torch.Tensor:
    # insole_start_time format is "12.07.2024 14:28:12.306"
    # sync_data format is "2024-07-12T14:29:07.242667"
    insole_start_datetime = datetime.strptime(insole_start_time, "%d.%m.%Y %H:%M:%S.%f")
    sync_datetime = datetime.strptime(sync_data, "%Y-%m-%dT%H:%M:%S.%f")

    assert sync_datetime > insole_start_datetime, "Sync data timestamp must be later than insole start time"

    # Calculate time difference in seconds
    time_diff = (sync_datetime - insole_start_datetime).total_seconds()

    # Calculate the number of frames to trim from the beginning
    delta_time_insole = 1.0 / args.insole_hz  # in seconds
    frames_to_trim = int(time_diff / delta_time_insole)

    # Trim the insole_data tensor
    trimmed_insole_data = insole_data[frames_to_trim:, ...]

    return trimmed_insole_data


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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name", type=str, help="Name of the dataset")
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--insole_hz", type=float, default=100.0)
    parser.add_argument("--pose_hz", type=float, default=25.0)
    parser.add_argument("--target_hz", type=float, default=30.0)
    parser.add_argument("--acc_world", action="store_true", default=True)
    parser.add_argument("--xsens", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"Processing {args.name} dataset -----------------")

    main(args)
