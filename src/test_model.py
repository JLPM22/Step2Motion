import subprocess
import torch

from dataset import MotionDataset
from argparse import ArgumentParser, Namespace


def main(args: Namespace) -> None:
    assert args.dataset.endswith("_test.pt"), "The dataset must be a test set"

    datasets = [args.dataset]
    if not args.only_test:
        datasets.append(args.dataset.replace("_test.pt", "_val.pt"))
        datasets.append(args.dataset.replace("_test.pt", "_train.pt"))

    for dataset_path in datasets:
        dataset = MotionDataset.load(dataset_path, torch.device("cpu"))
        n_clips = len(dataset.clips) - 1
        print(f"Dataset: {dataset_path} - Clips: {n_clips}")
        for i in range(n_clips):
            cmd_args = [
                "src/test.py",
                args.model,
                args.template_bvh,
                "--dataset",
                dataset_path,
                "--clip",
                str(i),
            ]
            print(" ".join(cmd_args))
            subprocess.run(["env/Scripts/python.exe"] + cmd_args)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the model folder")
    parser.add_argument("dataset", type=str, help="Path to the test dataset")
    parser.add_argument("template_bvh", type=str, help="Path to the template BVH file")
    parser.add_argument("--only_test", action="store_true", help="Only use the test set")
    args = parser.parse_args()

    main(args)
