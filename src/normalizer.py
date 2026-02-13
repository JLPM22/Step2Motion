import torch
import warnings
from argparse import ArgumentParser, Namespace
from dataset import MotionDataset


class Normalizer:
    def __init__(self, db: MotionDataset):
        poses = torch.cat(prior_db.poses, dim=0)  # type: ignore
        insoles = torch.cat(insole_db.insole, dim=0)  # type: ignore

        def compute_mean_std(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mean = data.mean(dim=0).float()
            std = data.std(dim=0)
            std = torch.where(std < 1e-6, torch.ones_like(std), std).float()
            return mean, std

        self.poses_mean, self.poses_std = compute_mean_std(poses)
        self.distances_mean, self.distances_std = compute_mean_std(db.distances)
        self.insoles_mean, self.insoles_std = compute_mean_std(insoles)

        self.id = hash(
            self.poses_mean.mean().item()
            + self.poses_std.mean().item()
            + self.distances_mean.mean().item()
            + self.distances_std.mean().item()
            + self.insoles_mean.mean().item()
            + self.insoles_std.mean().item()
        )

    def is_valid(self, id: int) -> bool:
        return self.id == id

    def normalize_poses(self, poses: torch.Tensor) -> torch.Tensor:
        return (poses - self.poses_mean) / self.poses_std

    def normalize_distances(self, distances: torch.Tensor) -> torch.Tensor:
        return (distances - self.distances_mean) / self.distances_std

    def normalize_insole(self, insoles: torch.Tensor) -> torch.Tensor:
        return (insoles - self.insoles_mean) / self.insoles_std

    def denormalize_poses(self, poses: torch.Tensor) -> torch.Tensor:
        return poses * self.poses_std + self.poses_mean

    def denormalize_distances(self, distances: torch.Tensor) -> torch.Tensor:
        return distances * self.distances_std + self.distances_mean

    def denormalize_insole(self, insoles: torch.Tensor) -> torch.Tensor:
        return insoles * self.insoles_std + self.insoles_mean

    def to(self, device: torch.device) -> None:
        self.poses_mean = self.poses_mean.to(device)
        self.poses_std = self.poses_std.to(device)
        self.distances_mean = self.distances_mean.to(device)
        self.distances_std = self.distances_std.to(device)
        self.insoles_mean = self.insoles_mean.to(device)
        self.insoles_std = self.insoles_std.to(device)


def main(args: Namespace) -> None:
    db = MotionDataset.load(args.db, torch.device("cpu"))
    normalizer = Normalizer(db)
    torch.save(normalizer, f"./configs/normalizer_{args.name}.pth")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = ArgumentParser()
    parser.add_argument("name", type=str, help="Name of the normalizer")
    parser.add_argument("db", type=str, help="Path to the dataset containing the control net data (insole)")
    args = parser.parse_args()

    main(args)
