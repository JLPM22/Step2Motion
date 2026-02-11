import json
import os
import random
import numpy as np
import torch
import pymotion.rotations.quat_torch as quat
from collections import defaultdict
from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple
from pymotion.io.bvh import BVH
from pymotion.ops.forward_kinematics import fk
from dataset import MotionDataset
from utils import skeleton_pos_to_rot


def find_files(
    pose_type: str, input_dir: str, file_basename: str
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    files = defaultdict(lambda: ["", ""])
    subjects = defaultdict(lambda: [])
    subjects_res = defaultdict(lambda: [])
    for dir in os.listdir(input_dir):
        if not os.path.isdir(os.path.join(input_dir, dir)):
            continue
        name = dir
        if "_bis" in dir or "_tris" in dir:
            name = dir.split("_")[0]
        subjects[name].append(dir)

    for subject, dirs in subjects.items():
        print(f"Reading Subject {subject}")
        for dir in dirs:
            print(f"    {dir}")
            path = os.path.join(input_dir, dir, "actions")
            for action in os.listdir(path):
                base_path = os.path.join(path, action)
                pose_path = os.path.join(base_path, "body_tracking", file_basename + pose_type)
                assert os.path.exists(pose_path), f"Path {pose_path} does not exist"
                insole_path = os.path.join(base_path, "insoles", "insole_reading.npz")
                assert os.path.exists(insole_path), f"Path {insole_path} does not exist"
                files[dir + "_" + action][0] = pose_path
                files[dir + "_" + action][1] = insole_path
                subjects_res[subject].append(base_path)
    return files, subjects_res


def generate_bvh(input_dir: str) -> None:
    _, subjects = find_files(".npz", input_dir, "joints_3D")
    for _, base_paths in subjects.items():
        offsets = None
        for base_path in base_paths:
            pose_path = os.path.join(base_path, "body_tracking", "joints_3D.npz")
            poses = np.load(pose_path)["translations"]
            if offsets is None:
                bvh = BVH()
                bvh.load(args.template_bvh)
                bvh.set_scale(0.0001)
                end_effectors = [8, 16, 20, 24, 28, 32, 33, 41, 45, 49, 53, 57, 58, 63, 68]
                joints_1 = [i for i in range(poses.shape[1]) if i not in end_effectors]
                poses = poses[:, joints_1]
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
                joints_2 = [i for i in range(poses.shape[1]) if i not in remove_joints]
                poses = poses[:, joints_2]
                bvh.remove_joints(remove_joints)
                parents = bvh.data["parents"]
                # get offsets form 1st frame
                offsets = poses[0, :] - poses[0, parents]
                bvh.data["offsets"] = offsets.copy()
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
                bvh.set_order_joints(reorder_joints)
                reverse_order = [reorder_joints.index(i) for i in range(len(reorder_joints))]
            else:
                poses = poses[:, joints_1]
                poses = poses[:, joints_2]
            # IK to get rotations
            rots = skeleton_pos_to_rot(poses, parents, offsets)
            # create bvh
            positions = np.tile(offsets, (poses.shape[0], 1, 1))
            positions[:, 0, :] = poses[:, 0]  # global position
            rots = rots[:, reverse_order]
            positions = positions[:, reverse_order]
            bvh.set_data(rots, positions)
            bvh.save(pose_path[: -len(".npz")] + ".bvh")


def process_pose_file(
    path: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bvh = BVH()
    bvh.load(path)
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


def process_files(
    input_dir: str, db_name: str, only_subject: Optional[str], only_subject_val: Optional[str]
) -> None:
    files, subjects = find_files(".bvh", input_dir, "joints_3D_xsens")

    if only_subject is None:
        subject_names = list(subjects.keys())
        random.shuffle(subject_names)

        num_subjects = len(subject_names)
        num_train = int(num_subjects * 0.8)
        num_val = int(num_subjects * 0.1)
        num_test = num_subjects - num_train - num_val
        train_subjects = subject_names[:num_train]
        val_subjects = subject_names[num_train : num_train + num_val]
        test_subjects = subject_names[num_train + num_val :]
    else:
        subject_names = [only_subject]
        if only_subject_val is not None:
            subject_names.append(only_subject_val)
        num_subjects = 1
        num_train = 1
        num_val = 1
        num_test = 1
        train_subjects = [only_subject]
        val_subjects = [only_subject_val if only_subject_val is not None else only_subject]
        test_subjects = [only_subject_val if only_subject_val is not None else only_subject]
    print(f"Number of subjects for - Train: {num_train}, Val: {num_val}, Test: {num_test}")
    print("Train subjects: ", train_subjects)
    print("Val subjects: ", val_subjects)
    print("Test subjects: ", test_subjects)
    with open(os.path.join(input_dir, f"{db_name}_split.json"), "w") as f:
        json.dump({"train": train_subjects, "val": val_subjects, "test": test_subjects}, f)

    # Process the data
    files_data = {}
    for key, [pose_path, insole_path] in files.items():
        insole = np.load(insole_path, allow_pickle=True)

        def get_data(data: dict) -> np.ndarray:
            pressure = data["pressure"]
            acc = data["acceleration"]
            angular_vel = data["angular_vel"]
            force = data["force"]
            cop = data["cop"]
            return np.concatenate([pressure, acc, angular_vel, force, cop], axis=-1)

        left_data = get_data(insole["left"].item())
        right_data = get_data(insole["right"].item())
        insole_data = np.concatenate([left_data, right_data], axis=-1)
        insole_data = torch.from_numpy(insole_data).float()
        pose_data, quats, parents, global_pos, global_rot, offsets = process_pose_file(pose_path)
        pose_data = pose_data[:-1]  # remove last frame to match insole data
        quats = quats[:-1]
        global_pos = global_pos[:-1]
        global_rot = global_rot[:-1]

        displacements = torch.cat(
            [
                torch.tensor([[0.0, 0.0, 0.0]], device=global_pos.device),
                global_pos[1:] - global_pos[:-1],
            ],
            dim=0,
        )

        files_data[key] = (insole_data, pose_data, quats, displacements, parents, offsets, global_rot)

    # Create the datasets
    def create_dataset(subjects: list[str], name: str) -> MotionDataset:  # type: ignore
        data = []
        for file, file_data in files_data.items():
            if any([s in file for s in subjects]):
                data.append(file_data)
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
            target_sample_rate=30.0,
            foot_indices=[7, 8, 3, 4],
        )
        print(f"{name} - Insole: {insole.shape}, Pose: {pose.shape}")
        torch.save(dataset, os.path.join(args.input_dir, f"{args.name}_{name}.pt"))
        return dataset

    if len(train_subjects) > 0:
        create_dataset(train_subjects, "train")
    if len(val_subjects) > 0:
        create_dataset(val_subjects, "val")
    if len(test_subjects) > 0:
        create_dataset(test_subjects, "test")


def main(args: Namespace) -> None:

    # IF generate_bvh is True, THEN GENERATE BVH WITH THE TOPOLOGY OF XSENS
    # then this can be retargeted with Blender
    # to be retargeted in Blender we need all of them to have a proper T-Pose reference frame
    # 1. Get the first pose of the first animation and manually adjust it to T-Pose
    # 2. Save that pose in the first keyframe of the animation and save it as subject_tpose.bvh
    # 3. A correct feet orientation is crucial for world acceleration to work properly.
    #    Import in blender the target armature (e.g., xsens S01) and use the Interactive Tweaks of ARP
    #    to adjust the feet until it looks good when retargeted (properly oriented, up, forward, etc.)
    #    Remember to save the remap_preset for the subject (i.e., subject_remap.bmap)
    # 4. Have a Blender script that copies the first keyframe of the subject_tpose.bvh to the reference pose of all animations of that subject
    # 5. Apply the retargeting script as always, with the subject-specific target armature computed in step 3
    # Then it can be processed without generate_bvh to create the dataset

    if args.generate_bvh:
        generate_bvh(args.input_dir)
    else:
        process_files(args.input_dir, args.name, args.subject, args.subject_val)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name", type=str, help="Name of the dataset")
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--acc_world", action="store_true", default=True)
    parser.add_argument("--generate_bvh", action="store_true", default=False)
    parser.add_argument("--template_bvh", type=str, default="")
    parser.add_argument("--subject", type=str, default=None, help="Process only one subject")
    parser.add_argument(
        "--subject_val", type=str, default=None, help="Process only one subject for validation"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"Processing {args.name} dataset -----------------")

    main(args)
