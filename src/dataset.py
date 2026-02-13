import torch
import pymotion.rotations.quat_torch as quat
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from pymotion.ops.skeleton_torch import fk


class MotionDataset(Dataset):

    def __init__(
        self,
        clips: list[int],
        poses: torch.Tensor,
        quats: torch.Tensor,
        displacements: torch.Tensor,
        global_rots: torch.Tensor,
        insole: torch.Tensor,
        parents: torch.Tensor,
        offsets: torch.Tensor,
        is_acceleration_world: bool,
        target_sample_rate: float,
        foot_indices: list[int],  # right foot, right toe, left foot, left toe
    ) -> None:
        """
        Args:
            clips (list[int]): list of clip start indices of length C
            poses (torch.Tensor): (N, (J - 1) x 3) tensor of poses minus the root joint
                J is the number of joints
            quats (torch.Tensor): (N, J x 4) tensor of quaternions
            displacements (torch.Tensor): (N, 3) tensor of displacements w.r.t to previous pose
            global_rots (torch.Tensor): (N, 4) tensor of global rotations w.r.t. to previous pose
            insole (torch.Tensor): (N, 2 x 25) tensor of insole data
                insole data is a concatenation of left and right insole data
                each foot has: 16 pressure + 3 acceleration IMU + 3 angular acceleration IMU + 1 total force + 2 center of pressure
            parents (torch.Tensor): (J,) tensor of parent indices
            offsets (torch.Tensor): (C, J, 3) tensor of joint offsets
            is_acceleration_world (bool): whether the acceleration data is in world space or in IMU space
        """
        self.temporality = 1
        self.poses = torch.cat([displacements, poses], dim=-1).float()
        self.insole = insole.float()
        self.parents = parents
        self.offsets = offsets
        self.quats = quats
        self.is_acceleration_world = is_acceleration_world
        self.quat_lIMU = None  # stored for visualization
        self.quat_rIMU = None  # stored for visualization
        self.initial_global_rot = global_rots[0]
        self.sample_rate = target_sample_rate  # Hz
        self.delta_time = 1.0 / self.sample_rate

        # compute distances
        self.distances = torch.linalg.vector_norm(offsets[:, 1:], dim=-1).type(torch.float32)

        # indices are [inclusive, exclusive)
        self.l_pressure_idx = (0, 16)
        self.l_acceleration_idx = (16, 19)
        self.l_angular_velocity_idx = (19, 22)
        self.l_total_force_idx = (22, 23)
        self.l_center_of_pressure_idx = (23, 25)
        self.r_pressure_idx = (25, 41)
        self.r_acceleration_idx = (41, 44)
        self.r_angular_velocity_idx = (44, 47)
        self.r_total_force_idx = (47, 48)
        self.r_center_of_pressure_idx = (48, 50)

        # compute local axes of the feet
        # assuming feet are symmetric in the rest pose
        assert len(foot_indices) == 4, "foot_indices must have 4 elements"
        self.rfoot_index = foot_indices[0]
        self.rtoes_index = foot_indices[1]
        self.lfoot_index = foot_indices[2]
        self.ltoes_index = foot_indices[3]
        pos, _ = fk(
            torch.tile(torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]), (1, parents.shape[0], 1)),
            torch.zeros((1, 3)).float(),
            offsets[0],
            parents,
        )

        def compute_axes(toes_index, foot_index, opposite_foot_index, invert_up=False):
            right = pos[:, toes_index] - pos[:, foot_index]
            right = right / torch.norm(right, dim=-1, keepdim=True)
            up = pos[:, opposite_foot_index] - pos[:, foot_index]
            if invert_up:
                up = -up
            up = up / torch.norm(up, dim=-1, keepdim=True)
            forward = torch.linalg.cross(right, up)
            forward = forward / torch.norm(forward, dim=-1, keepdim=True)
            up = torch.linalg.cross(forward, right)
            up = up / torch.norm(up, dim=-1, keepdim=True)
            return right, up, forward

        # rotations are zero in rest pose, thus axes are the same in local or world space at this point
        self.right_lfoot_local, self.up_lfoot_local, self.forward_lfoot_local = compute_axes(
            self.ltoes_index, self.lfoot_index, self.rfoot_index, invert_up=True
        )
        self.right_rfoot_local, self.up_rfoot_local, self.forward_rfoot_local = compute_axes(
            self.rtoes_index, self.rfoot_index, self.lfoot_index
        )

        assert is_acceleration_world, "Unless this is a test, this should be True"
        if is_acceleration_world:
            # Compute world space axes of the feet
            def compute_world_axes_foot(quats_foot, right, up, fwd):
                right_world = quat.mul_vec(quats_foot, right)
                right_world = right_world / torch.linalg.norm(right_world, dim=-1, keepdim=True)
                up_world = quat.mul_vec(quats_foot, up)
                up_world = up_world / torch.linalg.norm(up_world, dim=-1, keepdim=True)
                fwd_world = quat.mul_vec(quats_foot, fwd)
                fwd_world = fwd_world / torch.linalg.norm(fwd_world, dim=-1, keepdim=True)
                rotmat_IMU = matrix_from_basis_vectors(right_world, up_world, fwd_world)
                return quat.from_matrix(rotmat_IMU)

            self.quat_lIMU = compute_world_axes_foot(
                quats[:, self.lfoot_index],
                self.right_lfoot_local,
                self.up_lfoot_local,
                self.forward_lfoot_local,
            )
            self.quat_rIMU = compute_world_axes_foot(
                quats[:, self.rfoot_index],
                self.right_rfoot_local,
                self.up_rfoot_local,
                self.forward_rfoot_local,
            )

            left_acceleration = insole[:, slice(*self.l_acceleration_idx)]
            right_acceleration = insole[:, slice(*self.r_acceleration_idx)]
            self.left_acceleration_local = left_acceleration.clone()
            self.right_acceleration_local = right_acceleration.clone()
            # flip local Y-axis for the left foot (it's inverted in the IMU)
            left_acceleration[..., 1] = -left_acceleration[..., 1]
            left_acceleration_world = quat.mul_vec(self.quat_lIMU, left_acceleration)
            left_acceleration_world[..., 1] = left_acceleration_world[..., 1] - 1  # remove gravity
            right_acceleration_world = quat.mul_vec(self.quat_rIMU, right_acceleration)
            right_acceleration_world[..., 1] = right_acceleration_world[..., 1] - 1  # remove gravity
            # store acceleration in world space
            insole[:, slice(*self.l_acceleration_idx)] = left_acceleration_world
            insole[:, slice(*self.r_acceleration_idx)] = right_acceleration_world

        # transform to clip-based dataset
        poses_list = []
        insole_list = []
        left_acceleration_local_list = [] if self.left_acceleration_local is not None else None
        right_acceleration_local_list = [] if self.right_acceleration_local is not None else None
        quats_list = []
        quat_lIMU_list = [] if self.quat_lIMU is not None else None
        quat_rIMU_list = [] if self.quat_rIMU is not None else None
        self.n_poses = 0
        clips.append(len(poses))
        for clip_idx in range(len(clips) - 1):
            start_idx = clips[clip_idx]
            end_idx = clips[clip_idx + 1]
            # discard first frame (wrong displacement and incremental global rotation)
            poses_list.append(self.poses[start_idx + 1 : end_idx])
            insole_list.append(self.insole[start_idx + 1 : end_idx])
            if left_acceleration_local_list is not None and self.left_acceleration_local is not None:
                left_acceleration_local_list.append(self.left_acceleration_local[start_idx + 1 : end_idx])
            if right_acceleration_local_list is not None and self.right_acceleration_local is not None:
                right_acceleration_local_list.append(self.right_acceleration_local[start_idx + 1 : end_idx])
            quats_list.append(self.quats[start_idx + 1 : end_idx])
            if quat_lIMU_list is not None and self.quat_lIMU is not None:
                quat_lIMU_list.append(self.quat_lIMU[start_idx + 1 : end_idx])
            if quat_rIMU_list is not None and self.quat_rIMU is not None:
                quat_rIMU_list.append(self.quat_rIMU[start_idx + 1 : end_idx])
            self.n_poses += len(poses_list[-1])
            clips[clip_idx] -= clip_idx  # remove the discarded frames
        clips[-1] = self.n_poses
        self.clips = clips
        self.poses = poses_list
        self.insole = insole_list
        self.left_acceleration_local = left_acceleration_local_list
        self.right_acceleration_local = right_acceleration_local_list
        self.quats = quats_list
        self.quat_lIMU = quat_lIMU_list
        self.quat_rIMU = quat_rIMU_list

    def __len__(self) -> int:
        return self.n_poses // self.stride

    def __getitem__(self, start_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            idx (int): index of the sample

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: pose, insole, parents
            pose (torch.Tensor):
                if is_acceleration_world:
                    (T, 3 + (J - 1) x 3) tensor of displacement + pose (root joint is not included in the pose)
                J is the number of joints
                T is 1 by default, use self.add_temporality(T) to change its value
            insole (torch.Tensor): (T, 2 x 25) tensor of insole data
                insole data is a concatenation of left and right insole data
                each foot has: 16 pressure + 3 acceleration IMU + 3 angular acceleration IMU + 1 total force + 2 center of pressure
                T is 1 by default, use self.add_temporality(T) to change its value
            distances (torch.Tensor): (J - 1,) tensor of joint distances (except root joint)
            parents (torch.Tensor): (J,) tensor of parent indices
                J is the number of joints
        """
        start_idx *= self.stride
        clip_idx = 0
        for clip_start_idx in self.clips[1:]:
            if clip_start_idx > start_idx:
                break
            clip_idx += 1
        clip_count = self.clips[clip_idx + 1] - self.clips[clip_idx]
        start_idx -= self.clips[clip_idx]
        end_idx = start_idx + self.temporality
        if end_idx > clip_count:
            start_idx -= end_idx - clip_count
            end_idx = clip_count
        slice_idx = slice(start_idx, end_idx)
        return (
            self.poses[clip_idx][slice_idx],
            self.insole[clip_idx][slice_idx],
            self.distances[clip_idx],
            self.parents,
        )

    def set_clip(self, clip: int) -> None:
        self.poses = self.poses[clip : clip + 1]
        self.insole = self.insole[clip : clip + 1]
        if hasattr(self, "left_acceleration_local"):
            self.left_acceleration_local = self.left_acceleration_local[clip : clip + 1]
            self.right_acceleration_local = self.right_acceleration_local[clip : clip + 1]
        self.quat_lIMU = self.quat_lIMU[clip : clip + 1] if self.quat_lIMU is not None else None
        self.quat_rIMU = self.quat_rIMU[clip : clip + 1] if self.quat_rIMU is not None else None
        start_clip = self.clips[clip]
        end_clip = self.clips[clip + 1]
        self.clips = self.clips[clip : clip + 2]
        self.clips[0] -= start_clip
        self.clips[1] -= start_clip
        self.n_poses = end_clip - start_clip

    def set_temporality(self, temporality: int) -> None:
        self.temporality = temporality

    def set_stride(self, stride: int) -> None:
        self.stride = stride

    def to(self, device: torch.device) -> Dataset:
        for i in range(len(self.poses)):
            self.poses[i] = self.poses[i].to(device)
        for i in range(len(self.insole)):
            self.insole[i] = self.insole[i].to(device)
        for i in range(len(self.quats)):
            self.quats[i] = self.quats[i].to(device)
        self.parents = self.parents.to(device)
        self.distances = self.distances.to(device)
        self.right_lfoot_local = self.right_lfoot_local.to(device)
        self.up_lfoot_local = self.up_lfoot_local.to(device)
        self.forward_lfoot_local = self.forward_lfoot_local.to(device)
        self.right_rfoot_local = self.right_rfoot_local.to(device)
        self.up_rfoot_local = self.up_rfoot_local.to(device)
        self.forward_rfoot_local = self.forward_rfoot_local.to(device)
        self.initial_global_rot = self.initial_global_rot.to(device)
        return self

    def to_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def load(
        data_path: str,
        device: torch.device,
    ):
        return torch.load(data_path, weights_only=False).to(device)


def matrix_from_basis_vectors(right: torch.Tensor, up: torch.Tensor, forward: torch.Tensor) -> torch.Tensor:
    """
    Given three orthogonal basis vectors, create the rotation matrix that indicates the rotation from the coordinate
    system defined by the basis vectors to the coordinate system where those vectors are defined.
    Parameters
    ----------
        right: torch.Tensor[..., 3]. Right vector of the new coordinate system.
        up: torch.Tensor[..., 3]. Up vector of the new coordinate system.
        forward: torch.Tensor[..., 3]. Forward vector of the new coordinate system.
    Returns
    -------
        rotmats: np.array[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                     [r1.x, r1.y, r1.z],
                                                     [r2.x, r2.y, r2.z]] where ri is row i.
    """
    # Check if vectors are orthogonal and normalized (within a tolerance)
    assert torch.allclose(
        torch.abs(torch.sum(torch.linalg.cross(right, up) * forward, dim=-1)),
        torch.ones_like(forward[..., 0]),
        atol=1e-6,
    ), "Vectors are not orthogonal"
    assert torch.allclose(
        torch.abs(torch.sum(torch.linalg.cross(up, forward) * right, dim=-1)),
        torch.ones_like(right[..., 0]),
        atol=1e-6,
    ), "Vectors are not orthogonal"
    assert torch.allclose(
        torch.abs(torch.sum(torch.linalg.cross(forward, right) * up, dim=-1)),
        torch.ones_like(up[..., 0]),
        atol=1e-6,
    ), "Vectors are not orthogonal"
    # instead of comparing vectors, compare if the dotproduct of the cross result and the vector is close to 0
    assert torch.allclose(
        right.norm(dim=-1), torch.ones_like(right[..., 0]), atol=1e-6
    ), "Right vector is not normalized"
    assert torch.allclose(
        up.norm(dim=-1), torch.ones_like(up[..., 0]), atol=1e-6
    ), "Up vector is not normalized"
    assert torch.allclose(
        forward.norm(dim=-1), torch.ones_like(forward[..., 0]), atol=1e-6
    ), "Forward vector is not normalized"

    # Directly construct the rotation matrix using the basis vectors as columns
    rotmats = torch.stack([right, up, forward], dim=-1)

    return rotmats
