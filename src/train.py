import os
import argparse
import torch
import numpy as np
import random
import shutil
import warnings
from config import Config, PredictionMode, copy_config, load_config
from torch.utils.data import DataLoader
from dataset import MotionDataset
from normalizer import Normalizer
from utils import data_augmentation
from losses import PoseLoss, TranslationLoss
from backward_diffusion import ControlTransformer, PriorTransformer, TransformerTranslation, model_from_config
from metrics import validate_pose, validate_translation
from forward_diffusion import ForwardDiffusion


def train(
    train_loader: DataLoader,
    train_dataset: MotionDataset,
    val_loader: DataLoader,
    normalizer: Normalizer,
    forward_diffusion: ForwardDiffusion,
    backward_diffusion_pose: ControlTransformer,
    backward_diffusion_pose_prior: PriorTransformer,
    model_trans: TransformerTranslation,
    optimizer_pose: torch.optim.Optimizer,  # type: ignore
    optimizer_trans: torch.optim.Optimizer,  # type: ignore
    loss_fn: torch.nn.Module,
    loss_fn_trans: torch.nn.Module,
    config: Config,
    train_only_translation: bool,
    device: torch.device,
) -> None:
    T = forward_diffusion.T
    best_val_loss = float("inf")

    # Train pose diffusion model
    if not train_only_translation:
        print("Training pose diffusion model -----------------")
        for epoch in range(config["epochs_pose"]):
            train_loss = 0
            for step, (x_0, c, distances, parents) in enumerate(train_loader):
                optimizer_pose.zero_grad()

                x_0, c = data_augmentation(x_0, c, train_dataset)

                x_0 = normalizer.normalize_poses(x_0)
                c = normalizer.normalize_insole(c)
                distances = normalizer.normalize_distances(distances)

                x_0 = x_0[..., 3:]  # Keep only pose
                batch_size = x_0.shape[0]

                t = torch.randint(0, T + 1, (batch_size,), device=device)  # t in [0, T]
                z_uncond = torch.rand(batch_size)
                c_mask = (z_uncond > config["p_uncond"]).float().to(device)

                x_t, noise = forward_diffusion(x_0, t)
                prediction, _ = backward_diffusion_pose_prior(
                    x=x_t, distances=distances, t=t, control=backward_diffusion_pose, c=c, c_mask=c_mask
                )

                if config["prediction_mode"] == PredictionMode.X0:
                    loss = loss_fn(prediction, x_0, c)
                else:
                    raise ValueError(f"Unknown prediction mode: {config['prediction_mode']}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    backward_diffusion_pose_prior.parameters(), config["clip_grad_value"]
                )
                torch.nn.utils.clip_grad_norm_(
                    backward_diffusion_pose.parameters(), config["clip_grad_value"]
                )
                optimizer_pose.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss = validate_pose(
                val_loader,
                normalizer,
                forward_diffusion,
                backward_diffusion_pose,
                backward_diffusion_pose_prior,
                loss_fn,
                config,
                device,
            )
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                torch.save(
                    backward_diffusion_pose_prior.state_dict(),
                    os.path.join(config["model_dir"], "model_prior.pth"),
                )
                torch.save(
                    backward_diffusion_pose.state_dict(),
                    os.path.join(config["model_dir"], "model.pth"),
                )
            print(
                f"Epoch: {epoch}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f} - Best Val Loss: {best_val_loss:.5f}"
                + ("*" if is_best else "")
            )

    # Train translation model
    if model_trans is not None and optimizer_trans is not None:
        print("Training translation model -----------------")
        best_val_loss = float("inf")
        for epoch in range(config["epochs_trans"]):
            train_loss = 0
            for step, (x_0, c, distances, parents) in enumerate(train_loader):
                optimizer_trans.zero_grad()

                x_0, c = data_augmentation(x_0, c, train_dataset)

                x_0 = normalizer.normalize_poses(x_0)
                c = normalizer.normalize_insole(c)
                distances = normalizer.normalize_distances(distances)

                pose = x_0[..., 3:]
                translation = x_0[..., :3]
                batch_size = x_0.shape[0]

                prediction = model_trans(pose, c)

                loss = loss_fn_trans(prediction, translation)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_trans.parameters(), config["clip_grad_value"])
                optimizer_trans.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss = validate_translation(
                val_loader, normalizer, model_trans, loss_fn_trans, config, device
            )
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                torch.save(model_trans.state_dict(), os.path.join(config["model_dir"], "model_trans.pth"))
            print(
                f"Epoch: {epoch}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f} - Best Val Loss: {best_val_loss:.5f}"
                + ("*" if is_best else "")
            )


def main(config: Config, train_only_translation: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    normalizer: Normalizer = torch.load(
        f"./configs/normalizer_{config['normalizer']}.pth",
        weights_only=False,
    )
    normalizer.to(device)

    print("Setting up datasets -----------------")

    def create_dataloader(key: str, batch_size: int) -> tuple[DataLoader, MotionDataset]:
        dataset = MotionDataset.load(config[key], device)
        dataset.set_temporality(config["input_T"])
        dataset.set_stride(config["input_stride"])
        return dataset.to_dataloader(batch_size), dataset

    train_data_key = "train_data"
    val_data_key = "val_data"
    test_data_key = "test_data"
    train_loader, train_dataset = create_dataloader(train_data_key, config["batch_size"])
    val_loader, _ = create_dataloader(val_data_key, config["batch_size"])
    _, test_dataset = create_dataloader(test_data_key, config["batch_size"])

    print("Setting up model -----------------")
    forward_diffusion = ForwardDiffusion(T=config["diffusion_T"]).to(device)
    bw_diff_pose_prior, _ = model_from_config(config, is_prior=True)
    bw_diff_pose_prior.to(device)
    bw_diff_pose, model_trans = model_from_config(config, is_prior=False)
    assert model_trans is not None
    bw_diff_pose.to(device)
    model_trans.to(device)

    if bw_diff_pose_prior.normalizer_id.item() == 0:
        bw_diff_pose_prior.set_normalizer_id(normalizer.id)
    assert bw_diff_pose_prior.normalizer_id.item() == normalizer.id
    if bw_diff_pose.normalizer_id.item() == 0:
        bw_diff_pose.set_normalizer_id(normalizer.id)
    if model_trans.normalizer_id.item() == 0:
        model_trans.set_normalizer_id(normalizer.id)
    assert bw_diff_pose.normalizer_id.item() == normalizer.id
    assert model_trans.normalizer_id.item() == normalizer.id

    pose_prior_params = {sum(p.numel() for p in bw_diff_pose_prior.parameters())}
    print("Prior pose model parameters: ", pose_prior_params)
    pose_params = {sum(p.numel() for p in bw_diff_pose.parameters())}
    trans_params = {sum(p.numel() for p in model_trans.parameters())}
    print("Pose model parameters: ", pose_params)
    print("Translation model parameters: ", trans_params)

    lr = config["lr"]
    # optimize both bw_diff_pose_prior and bw_diff_pose
    pose_params = list(bw_diff_pose_prior.parameters()) + list(bw_diff_pose.parameters())
    optimizer_pose = torch.optim.Adam(pose_params, lr=lr)  # type: ignore
    optimizer_trans = torch.optim.Adam(model_trans.parameters(), lr=lr)  # type: ignore
    loss_fn = PoseLoss(device)
    loss_fn_trans = TranslationLoss()

    print("Training the model -----------------")
    assert isinstance(bw_diff_pose_prior, PriorTransformer)
    assert isinstance(bw_diff_pose, ControlTransformer)
    assert isinstance(model_trans, TransformerTranslation)
    train(
        train_loader,
        train_dataset,
        val_loader,
        normalizer,
        forward_diffusion,
        bw_diff_pose,
        bw_diff_pose_prior,
        model_trans,
        optimizer_pose,
        optimizer_trans,
        loss_fn,
        loss_fn_trans,
        config,
        train_only_translation,
        device,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default=os.path.join("configs", "config.json")
    )
    parser.add_argument(
        "--only_translation", action="store_true", help="Train only the translation model", default=False
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    config["verbose"] = args.verbose

    config["model_dir"] = os.path.join(config["models_dir"], f"{config['name']}")
    copy_config(config, config["model_dir"], "copy_config.json")

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    main(config, args.only_translation)
