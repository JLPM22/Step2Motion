import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Parse category from command-line argument
if len(sys.argv) < 2:
    print("Usage: python visualize_metrics.py [up|dance|step2motion]")
    sys.exit(1)
category = sys.argv[1].lower()

# Modify only these variables
root_dir = os.path.join(".", "models")
if category == "up":
    datasets = ["underpressure_test"]
    clips = [21]
    models = ["UnderPressure"]
    names = ["Ours"]
elif category == "dance":
    datasets = ["dance_test"]
    clips = [2]
    models = ["dancing"]
    names = ["Ours"]
elif category == "step2motion":
    # Placeholder for 'ours' category
    datasets = ["step2motion_test"]
    clips = [1]
    models = ["step2motion"]
    names = ["Ours"]
else:
    raise ValueError("Invalid category")
selected_metrics = ["mpjpe", "mpjpe_legs", "mpjve_legs"]
assert len(models) == len(names)

# Do not modify below this line
models_dir = [os.path.join(root_dir, model, "predictions") for model in models]
metrics = ["mpjpe", "mpjpe_legs", "mpjve_legs"]
metrics_names = ["MPJPE", "MPJPE (Legs)", "MPJVE (Legs)"]


def load_data(model_dir: str) -> tuple[np.ndarray, float, float]:
    data = []
    for d_i, dataset in enumerate(datasets):
        for clip in range(clips[d_i]):
            data_path = os.path.join(model_dir, f"{dataset}_c{clip}_stats.npz")
            data.append(np.load(data_path)[metric].flatten())
    data = np.concatenate(data)
    mean = float(np.mean(data))
    std = float(np.std(data))
    return data, mean, std


for metric in metrics:
    if metric not in selected_metrics:
        continue

    plt.figure()
    for i, model_dir in enumerate(models_dir):
        data, mean, std = load_data(model_dir)
        sns.kdeplot(data, label=f"{names[i]} (Mean: {mean:.3f}, Std: {std:.3f})")
    plt.xlabel("Error (m)")
    plt.ylabel("Density")
    plt.title(f"{metrics_names[metrics.index(metric)]} Distribution")
    plt.legend()
    plt.show()

    plt.figure()
    for i, model_dir in enumerate(models_dir):
        data, mean, std = load_data(model_dir)
        plt.plot(data, label=f"{names[i]} (Mean: {mean:.3f}, Std: {std:.3f})", alpha=0.3)
    plt.xlabel("Frame")
    plt.ylabel("Error (m)")
    plt.title(f"{metrics_names[metrics.index(metric)]} Over Time")
    plt.legend()
    plt.show()
