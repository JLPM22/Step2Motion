import json
import os
from typing import TypedDict
from enum import Enum


class PredictionMode(Enum):
    NOISE = "noise"
    X0 = "x0"


class ModelType(Enum):
    TRANSFORMER = "transformer"


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value  # Serialize as the enum's string value
        return json.JSONEncoder.default(self, obj)  # Default behavior for other types


class Config(TypedDict):
    seed: int
    verbose: bool  # auto-generated
    name: str
    normalizer: str
    model_type: ModelType
    prior_train_data: str
    prior_val_data: str
    prior_test_data: str
    train_data: str
    val_data: str
    test_data: str
    models_dir: str
    model_dir: str  # auto-generated
    epochs_pose: int
    epochs_trans: int
    batch_size: int
    test_batch_size: int
    lr_prior: float
    lr: float
    clip_grad_value: float
    input_dim: int
    input_T: int
    input_stride: int
    output_dim: int
    hidden_dim: int
    num_layers: int
    # Diffusion specific
    prediction_mode: PredictionMode
    p_uncond: float
    diffusion_T: int
    diffusion_val_T: int
    # Transformer specific
    transformer_n_heads: int
    transformer_d_ff: int
    transformer_dropout: float


def copy_config(config: Config, path: str, name: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, name), "w") as f:
        json.dump(config, f, cls=EnumEncoder)


def load_config(path: str) -> Config:
    if not os.path.exists(path):
        raise ValueError(f"Config file {path} does not exist")

    with open(path, "r") as f:
        config = json.load(f)

    assert config["input_T"] % 2 == 0, "input_T must be divisible by 2"

    try:
        config["model_type"] = ModelType(config["model_type"])
    except ValueError:
        raise ValueError(f"Invalid data mode: {config['model_type']}. Valid options are: {list(ModelType)}")

    try:
        config["prediction_mode"] = PredictionMode(config["prediction_mode"])
    except ValueError:
        raise ValueError(
            f"Invalid data mode: {config['prediction_mode']}. Valid options are: {list(PredictionMode)}"
        )

    return config
