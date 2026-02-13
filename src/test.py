import json
import os
import random
import torch
import numpy as np
import warnings
from typing import Optional
from argparse import ArgumentParser
from config import Config, load_config
from dataset import MotionDataset
from dataset_utils import get_poses_from_data
from backward_diffusion import ControlTransformer, PriorTransformer, TransformerTranslation, model_from_config
from forward_diffusion import ForwardDiffusion
from losses import PoseLoss, TranslationLoss
from pymotion.render.viewer import Viewer
from metrics import get_metrics, test
from pymotion.io.bvh import BVH
from normalizer import Normalizer
from utils import skeleton_pos_to_rot


def pred_to_bvh(
    data: torch.Tensor,
    clip: int,
    dataset: MotionDataset,
    normalizer: Normalizer,
    out_path: str,
    template_bvh: str,
):
    """
    data is the ground truth tensor of shape (frames, output_dim)
    """
    data_np: np.ndarray = normalizer.denormalize_poses(data).cpu().numpy()
    displacements = data_np[:, :3]
    global_pos = np.cumsum(displacements, axis=0)
    poses = data_np[:, 3:].reshape(data_np.shape[0], -1, 3)
    poses = np.concatenate([np.zeros((data_np.shape[0], 1, 3)), poses], axis=1)
    parents = dataset.parents.cpu().numpy()
    offsets = dataset.offsets[clip].cpu().numpy()
    rots = skeleton_pos_to_rot(poses, parents, offsets)
    bvh = BVH()
    bvh.load(template_bvh)
    bvh.data["frame_time"] = dataset.delta_time
    positions = bvh.data["positions"]
    positions = np.tile(positions[0], (global_pos.shape[0], 1, 1))
    positions[:, 0, :] = global_pos
    bvh.set_data(rots, positions)
    bvh.save(out_path)


def cs_to_json(cs_in: torch.Tensor, dataset: MotionDataset, normalizer: Normalizer, out_path: str) -> None:
    """
    cs is the condition tensor of shape (frames, input_dim)
    """
    cs: np.ndarray = normalizer.denormalize_insole(cs_in).cpu().numpy()
    l_pressures = cs[:, slice(*dataset.l_pressure_idx)]
    l_accelerations = cs[:, slice(*dataset.l_acceleration_idx)]
    l_angular_accelerations = cs[:, slice(*dataset.l_angular_velocity_idx)]
    l_total_force = cs[:, slice(*dataset.l_total_force_idx)]
    l_center_of_pressure = cs[:, slice(*dataset.l_center_of_pressure_idx)]
    r_pressures = cs[:, slice(*dataset.r_pressure_idx)]
    r_accelerations = cs[:, slice(*dataset.r_acceleration_idx)]
    r_angular_accelerations = cs[:, slice(*dataset.r_angular_velocity_idx)]
    r_total_force = cs[:, slice(*dataset.r_total_force_idx)]
    r_center_of_pressure = cs[:, slice(*dataset.r_center_of_pressure_idx)]
    data = {
        "l_pressures": l_pressures.reshape(-1).tolist(),
        "l_accelerations": l_accelerations.reshape(-1).tolist(),
        "l_angular_accelerations": l_angular_accelerations.reshape(-1).tolist(),
        "l_total_force": l_total_force.reshape(-1).tolist(),
        "l_center_of_pressure": l_center_of_pressure.reshape(-1).tolist(),
        "r_pressures": r_pressures.reshape(-1).tolist(),
        "r_accelerations": r_accelerations.reshape(-1).tolist(),
        "r_angular_accelerations": r_angular_accelerations.reshape(-1).tolist(),
        "r_total_force": r_total_force.reshape(-1).tolist(),
        "r_center_of_pressure": r_center_of_pressure.reshape(-1).tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(data, f)


def attn_mat_to_json(
    attn_mats: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]], out_path: str
) -> None:
    """
    attn_mats is a tuple of 3 list of attention matrices of shape (n_heads, T//2 (left leg or right leg or body), T//2 (insole))
    """
    assert len(attn_mats) > 0, "No attention matrices to save"
    num_heads = attn_mats[0][0].shape[0]
    heads_left_leg = []
    heads_right_leg = []
    heads_body = []
    for head in range(num_heads):
        left_leg = []
        right_leg = []
        body = []
        for attn_mat in attn_mats[0]:
            left_leg.append(attn_mat[head].cpu().numpy())
        for attn_mat in attn_mats[1]:
            right_leg.append(attn_mat[head].cpu().numpy())
        for attn_mat in attn_mats[2]:
            body.append(attn_mat[head].cpu().numpy())
        heads_left_leg.append(np.concatenate(left_leg, axis=0))
        heads_right_leg.append(np.concatenate(right_leg, axis=0))
        heads_body.append(np.concatenate(body, axis=0))

    data = {}
    for i in range(num_heads):
        data[f"head_{i}_left_leg"] = heads_left_leg[i].flatten().tolist()
        data[f"head_{i}_right_leg"] = heads_right_leg[i].flatten().tolist()
        data[f"head_{i}_body"] = heads_body[i].flatten().tolist()

    with open(out_path, "w") as f:
        json.dump(data, f)


@torch.no_grad()
def main(
    config: Config,
    dataset_path: str,
    verbose: bool,
    template_bvh: str,
    w: float,
    clip: int,
    seed: Optional[int],
    vertical_zero_threshold: bool,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    normalizer: Normalizer = torch.load(
        f"./configs/normalizer_{config['normalizer']}.pth", weights_only=False
    )
    normalizer.to(device)

    print("Setting up models -----------------")
    forward_diffusion = ForwardDiffusion(T=config["diffusion_T"]).to(device)
    bw_diff_pose_prior, _ = model_from_config(config, is_prior=True)
    bw_diff_pose_prior.to(device)
    bw_diff_pose = None
    model_trans = None
    bw_diff_pose, model_trans = model_from_config(config, is_prior=False)
    assert model_trans is not None
    bw_diff_pose.to(device)
    model_trans.to(device)

    bw_prior_name = None
    bw_name = None
    trans_name = None
    bw_prior_name = "model_prior.pth"
    bw_name = "model.pth"
    trans_name = "model_trans.pth"

    if bw_prior_name is not None:
        bw_diff_pose_prior.load_state_dict(torch.load(os.path.join(config["model_dir"], bw_prior_name)))
        assert bw_diff_pose_prior.normalizer_id.item() == normalizer.id
    if bw_name is not None and bw_diff_pose is not None:
        bw_diff_pose.load_state_dict(torch.load(os.path.join(config["model_dir"], bw_name)))
        assert bw_diff_pose.normalizer_id.item() == normalizer.id
    if trans_name is not None and model_trans is not None:
        model_trans.load_state_dict(torch.load(os.path.join(config["model_dir"], trans_name)))
        assert model_trans.normalizer_id.item() == normalizer.id

    print("Setting up datasets -----------------")
    dataset = MotionDataset.load(dataset_path, device)
    dataset.set_clip(clip)
    dataset.set_temporality(config["input_T"])
    dataset.set_stride(1)

    print("Predicting -----------------")
    if bw_diff_pose is not None:
        assert isinstance(bw_diff_pose, ControlTransformer)
    assert isinstance(bw_diff_pose_prior, PriorTransformer)
    if model_trans is not None:
        assert isinstance(model_trans, TransformerTranslation)
    loss_fn = PoseLoss(device)
    loss_fn_trans = TranslationLoss()
    loss_pose, loss_trans, gt, pred, cs, _ = test(
        dataset,
        normalizer,
        forward_diffusion,
        bw_diff_pose,
        bw_diff_pose_prior,
        model_trans,
        loss_fn,
        loss_fn_trans,
        config,
        w,
        device,
        False,  # return_attn_mats
        vertical_zero_threshold,
    )
    print(f"Test Loss Pose: {loss_pose:.5f} - Test Loss Translation: {loss_trans:.5f}")
    mpjpe, mpeepe, mrpe, mpjpe_legs, mpjve_legs, mpeepe_legs = get_metrics(
        normalizer, gt, pred, dataset.sample_rate
    )

    def print_error(name: str, error: tuple[float, float, np.ndarray]) -> None:
        print(f"{name}: {error[0]:.5f} (std: {error[1]:.5f})")

    print_error("MPJPE", mpjpe)
    print_error("MPEEPE", mpeepe)
    print_error("MRPE", mrpe)
    print_error("MPJPE Legs", mpjpe_legs)
    print_error("MPJVE Legs", mpjve_legs)
    print_error("MPEEPE Legs", mpeepe_legs)

    print("Writing predictions -----------------")
    output_dir = os.path.join(config["model_dir"], "predictions")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    name = os.path.basename(dataset_path).split(".")[0] + "_c" + str(clip)
    pred_to_bvh(
        gt,
        clip,
        dataset,
        normalizer,
        os.path.join(output_dir, name + "_gt.bvh"),
        template_bvh,
    )
    pred_path = os.path.join(output_dir, name + "_pred.bvh")
    if seed is not None:
        pred_path = pred_path.replace(".bvh", f"_s{seed}.bvh")
    pred_to_bvh(
        pred,
        clip,
        dataset,
        normalizer,
        pred_path,
        template_bvh,
    )
    cs_to_json(
        cs,
        dataset,
        normalizer,
        os.path.join(output_dir, name + "_cs.json"),
    )
    np.savez(
        os.path.join(output_dir, name + "_stats.npz"),
        mpjpe=mpjpe[2],
        mpeepe=mpeepe[2],
        mrpe=mrpe[2],
        mpjpe_legs=mpjpe_legs[2],
        mpjve_legs=mpjve_legs[2],
        mpeepe_legs=mpeepe_legs[2],
    )

    if verbose:
        print("Visualizing predictions -----------------")
        max_frames = 5000
        parents = dataset.parents.cpu().numpy()
        viewer = Viewer(xy_size=8)
        viewer.add_floor(height=-1.0, size=8)

        def get_poses(poses):
            data = normalizer.denormalize_poses(poses).cpu().numpy()
            poses, _ = get_poses_from_data(
                data,
                add_global_pos=True,
            )
            poses[:, :, [1, 2]] = poses[:, :, [2, 1]]
            poses[..., 1] = -poses[..., 1]  # invert y axis
            return poses

        poses_gt = get_poses(gt)
        viewer.add_skeleton(poses_gt[:max_frames], parents)
        poses_pred = get_poses(pred)
        viewer.add_skeleton(poses_pred[:max_frames], parents, color="blue")
        viewer.run()


if __name__ == "__main__":
    DEFAULT_SEED = 2222

    warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")
    warnings.filterwarnings(
        "ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False"
    )
    warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
    warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention")

    parser = ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Path to the model directory")
    parser.add_argument("template_bvh", type=str, help="Path to the template BVH file")
    parser.add_argument(
        "--dataset", type=str, default="from_config_file", help="Path of the dataset to use for prediction"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print additional information during the process"
    )
    parser.add_argument(
        "--w",
        type=float,
        default=1.0,
        help="Classifier-Free Guidance weight (default: 1.0), it increases accuracy at the cost of diversity.",
    )
    parser.add_argument(
        "--clip",
        type=int,
        default=0,
        help="Clip to process (default: 0), useful when the dataset contains multiple clips",
    )
    parser.add_argument(
        "--vertical_th",
        action="store_true",
        default=False,
        help="Improve vertical prediction in some cases by usign a threshold to remove noise around zero",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed to use for random number generation (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.model_dir):
        raise ValueError(f"Model directory {args.model_dir} does not exist")

    config_path = os.path.join(args.model_dir, "copy_config.json")
    config = load_config(config_path)

    config["model_dir"] = args.model_dir

    dataset_path = args.dataset
    if args.dataset == "from_config_file":
        dataset_path = config["test_data"]

    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset {dataset_path} does not exist")

    main(
        config,
        dataset_path,
        args.verbose,
        args.template_bvh,
        args.w,
        args.clip,
        None if args.seed == DEFAULT_SEED else args.seed,
        args.vertical_th,
    )
