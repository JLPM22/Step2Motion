import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from dataset import MotionDataset
import pymotion.rotations.quat as quat
from pymotion.ops.skeleton import fk


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def data_augmentation(
    x_0: torch.Tensor, c: torch.Tensor, dataset: Optional[MotionDataset]
) -> tuple[torch.Tensor, torch.Tensor]:
    angle_y = torch.rand(1) * 2 * np.pi
    rot_y = torch.tensor(
        [[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]],
        device=x_0.device,
    ).float()

    x_0_shape = x_0.shape
    x_0 = x_0.view(x_0.shape[:-1] + (-1, 3))
    x_0 = x_0 @ rot_y
    x_0 = x_0.view(x_0_shape)

    if dataset is not None:
        c_lacc = c[..., slice(*dataset.l_acceleration_idx)]
        c_racc = c[..., slice(*dataset.r_acceleration_idx)]
        c_lacc = c_lacc @ rot_y
        c_racc = c_racc @ rot_y
        c[..., slice(*dataset.l_acceleration_idx)] = c_lacc
        c[..., slice(*dataset.r_acceleration_idx)] = c_racc

    return x_0, c


class SinusoidalPositionalEncoding(nn.Module):
    """
    Embedding[position, 2k] = sin(position / (n^(2k / d_model)))
    Embedding[position, 2k+1] = cos(position / (n^(2k / d_model)))
    where
        'position' is the position in the sequence
        'k' is the dimension index within the embedding vector
        'd_model' is the dimension of the embedding vector
        'n' increases the frequency of the oscillation, typically set to 10000
    """

    def __init__(self, max_length: int, n: int = 10000, d_model: int = 512, dropout: float = 0.1) -> None:
        assert d_model % 2 == 0, "d_model must be an even number"
        super(SinusoidalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)
        position = torch.arange(end=max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(n) / d_model))
        positional_encoding = torch.zeros(max_length, d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param t: Time step, shape (batch_size,)
        """
        return self.positional_encoding[t, :]

    def forward_temporality(self, m: int) -> torch.Tensor:
        """
        Forward pass
        :param m: Number of time steps
        """
        return self.positional_encoding[:m, :]


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize a vector

    Parameters
    ----------
    v : np.array
    eps : float
        A small epsilon to prevent division by zero.

    Returns
    -------
    normalized_v : np.array
    """
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + eps)


def from_to(v1: np.ndarray, v2: np.ndarray, normalize_input: bool = True) -> np.ndarray:
    """
    Calculate the quaternion that rotates direction v1 to direction v2,
    handling edge cases for parallel vectors individually.

    Parameters
    ----------
    v1, v2 : np.array[..., [x,y,z]]
        Input vectors representing directions.
    normalize_input : bool
        Whether to normalize the input vectors.

    Returns
    -------
    rot : np.array[..., [w,x,y,z]]
        Quaternion representing the rotation.
    """
    assert v1.shape[-1] == 3 and v2.shape[-1] == 3, "Input vectors must have shape [..., 3]"
    assert v1.shape == v2.shape, "Input vectors must have the same shape"

    if normalize_input:
        v1_norm = normalize(v1)
        v2_norm = normalize(v2)
    if v1.ndim == 1:
        v1_norm = v1_norm[np.newaxis, :]
        v2_norm = v2_norm[np.newaxis, :]

    # Calculate cross product and dot product
    cross = np.cross(v1_norm, v2_norm)
    dot = np.sum(v1_norm * v2_norm, axis=-1, keepdims=True)  # type: ignore

    # Handle general case using np.isclose for robust "near zero" detection
    axis_rot = normalize(cross)
    w = np.sqrt((1 + dot) * 0.5)  # cos(theta/2) = sqrt((1 + dot) / 2)
    s = np.sqrt((1 - dot) * 0.5)  # sin(theta/2) = sqrt((1 - dot) / 2)
    rot = np.concatenate([w, axis_rot * s], axis=-1)

    # Handle parallel vectors (dot ≈ 1)
    parallel = np.isclose(dot, 1.0).flatten()
    rot[parallel] = [1.0, 0.0, 0.0, 0.0]

    # Handle anti-parallel vectors (dot ≈ -1)
    anti_parallel = np.isclose(dot, -1.0).flatten()
    for i in np.where(anti_parallel)[0]:
        # Find a non-collinear vector
        if np.allclose(np.abs(v1_norm[i, 0]), 1.0):
            orthogonal = np.array([0.0, 1.0, 0.0])
        else:
            orthogonal = np.array([1.0, 0.0, 0.0])
        axis_rot = quat.normalize(np.cross(v1_norm[i], orthogonal))
        rot[i] = np.concatenate([[0.0], axis_rot])

    return rot


def from_to_axis(
    v1: np.ndarray, v2: np.ndarray, rot_axis: np.ndarray, normalize_input: bool = True
) -> np.ndarray:
    """
    Calculate the quaternion that rotates direction v1 to direction v2,
    handling edge cases for parallel vectors individually.
    The rotation axis is provided.

    Parameters
    ----------
    v1, v2 : np.array[..., [x,y,z]]
        Input vectors representing directions.
    axis : np.array[..., [x,y,z]]
    normalize_input : bool
        Whether to normalize the input vectors.

    Returns
    -------
    rot : np.array[..., [w,x,y,z]]
        Quaternion representing the rotation.
    """
    assert v1.shape[-1] == 3 and v2.shape[-1] == 3, "Input vectors must have shape [..., 3]"
    assert v1.shape == v2.shape, "Input vectors must have the same shape"

    if normalize_input:
        v1_norm = normalize(v1)
        v2_norm = normalize(v2)
    if v1.ndim == 1:
        v1_norm = v1_norm[np.newaxis, :]
        v2_norm = v2_norm[np.newaxis, :]

    # Calculate cross product and dot product
    cross = np.cross(v1_norm, v2_norm)
    dot = np.sum(v1_norm * v2_norm, axis=-1, keepdims=True)  # type: ignore

    # Handle general case using np.isclose for robust "near zero" detection
    w = np.sqrt((1 + dot) * 0.5)  # cos(theta/2) = sqrt((1 + dot) / 2)
    s = np.sqrt((1 - dot) * 0.5)  # sin(theta/2) = sqrt((1 - dot) / 2)
    # Adjust sign of s based on cross product and rot_axis
    cross_dot_axis = np.sum(cross * rot_axis, axis=-1, keepdims=True)
    s *= np.sign(cross_dot_axis)  # Correct sign based on alignment
    # Combine w and s to form quaternion
    rot = np.concatenate([w, rot_axis * s], axis=-1)

    # Handle parallel vectors (dot ≈ 1)
    parallel = np.isclose(dot, 1.0).flatten()
    rot[parallel] = [1.0, 0.0, 0.0, 0.0]

    # Handle anti-parallel vectors (dot ≈ -1)
    anti_parallel = np.isclose(dot, -1.0)
    anti_parallel_rotmask = np.tile(anti_parallel, (1,) * (rot.ndim - 1) + (4,))
    anti_parallel_rotaxismask = np.tile(anti_parallel, (1,) * (rot_axis.ndim - 1) + (3,))
    rots_anti_parallel = rot[anti_parallel_rotmask]
    rots_anti_parallel[::4] = 0
    rots_anti_parallel[[False, True, True, True] * (len(rots_anti_parallel) // 4)] = rot_axis[
        anti_parallel_rotaxismask
    ]
    rot[anti_parallel_rotmask] = rots_anti_parallel

    return rot


def skeleton_pos_to_rot(positions: np.ndarray, parents: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    positions: (frames, joints, 3)
    parents: (joints,)
    offsets: (joints, 3)
    ------------------------------
    result: (frames, joints, 4)
    """

    nFrames = positions.shape[0]
    nJoints = parents.shape[0]

    # Find all children for each joint:
    children = [[] for _ in range(nJoints)]
    for i, parent in enumerate(parents):
        if i > 0:  # Ensure valid parent index
            children[parent].append(i)

    # Iterate joints and align directions from the rest pose to the predicted pose
    rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (nFrames, nJoints, 1))
    for j, children_of_j in enumerate(children):
        if len(children_of_j) == 0:  # Skip joints with 0 children
            continue

        # Compute current pose (start with rest pose)
        pos, rotmats = fk(
            rotations,
            np.zeros((1, 3)),
            offsets,
            parents,
        )
        global_rots = quat.from_matrix(rotmats)
        # Align current pose to predicted pose based on the first child
        c = children_of_j[0]
        rest_dir = pos[:, c] - pos[:, j]
        rest_dir = quat.mul_vec(quat.inverse(global_rots[:, j]), rest_dir)
        pred_dir = positions[:, c] - positions[:, j]
        pred_dir = quat.mul_vec(quat.inverse(global_rots[:, j]), pred_dir)
        rot = from_to(rest_dir, pred_dir)
        rotations[:, j] = rot

        # If more than one child, use it for roll correction
        for gc in children_of_j[1:]:
            pos, rotmats = fk(
                rotations,
                np.zeros((1, 3)),
                offsets,
                parents,
            )
            global_rots = quat.from_matrix(rotmats)
            # Align
            rest_gc_dir = pos[:, gc] - pos[:, j]
            rest_gc_dir = quat.mul_vec(quat.inverse(global_rots[:, j]), rest_gc_dir)
            pred_gc_dir = positions[:, gc] - positions[:, j]
            pred_gc_dir = quat.mul_vec(quat.inverse(global_rots[:, j]), pred_gc_dir)
            roll_axis = quat.mul_vec(
                quat.inverse(global_rots[:, j]), normalize(positions[:, c] - positions[:, j])
            )
            roll_rot = from_to_axis(rest_gc_dir, pred_gc_dir, roll_axis)
            rotations[:, j] = quat.mul(rotations[:, j], roll_rot)

    return rotations
