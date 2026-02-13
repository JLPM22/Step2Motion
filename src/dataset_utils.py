import numpy as np


def get_poses_from_data(poses: np.ndarray, add_global_pos: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract poses from the data tensor.
    Parameters
    ----------
        poses : np.ndarray[N, J x 3]. Poses tensor.
        initial_global_rot : np.ndarray[4]. Initial global rotation.
        add_global_pos : bool. Whether to add global position to the poses.
    Returns
    -------
        poses: np.ndarray[N, J, 3]. Poses tensor.
        global_pos: np.ndarray[N, 3]. Global positions.
    """
    poses = poses.reshape((poses.shape[0], -1, 3))
    # [   0->1, 1->2, 2->3, ...] - displacement
    displacement = poses[:, 0:1, :]
    # [0, 0->1, 1->2, 2->3, ...] - displacement
    displacement = np.concatenate([np.zeros((1, 1, 3)), displacement[:-1]], axis=0)
    # [0, 1,    2,    3,    ...] - global_pos
    global_pos = np.cumsum(displacement, axis=0)
    # [   1,    2,    3,    ...] - poses
    poses = poses[:-1, 1:, :]
    # [   1,    2,    3,    ...] - global_rots & global_pos
    global_pos = global_pos[1:]
    if add_global_pos:
        poses = poses + global_pos
        poses = np.concatenate([global_pos, poses], axis=1)  # add root joint
    return poses, global_pos
