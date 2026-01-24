import os
import pickle
import numpy as np
import torch


def _check_paths(args):
    data_path = args["input"]["data path"]
    indices_path = args["input"]["indices"]

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"[Preflight] Dataset path not found: {data_path}"
        )

    if not os.path.exists(indices_path):
        raise FileNotFoundError(
            f"[Preflight] indices.pkl not found: {indices_path}"
        )

    print("✓ Dataset and indices paths exist")


def _check_dataset_structure(data_path):
    lenses_dir = os.path.join(data_path, "lenses")
    nonlenses_dir = os.path.join(data_path, "nonlenses")

    if not os.path.isdir(lenses_dir):
        raise FileNotFoundError(
            "[Preflight] Missing 'lenses/' directory inside data_root"
        )

    if not os.path.isdir(nonlenses_dir):
        raise FileNotFoundError(
            "[Preflight] Missing 'nonlenses/' directory inside data_root"
        )

    lens_files = [f for f in os.listdir(lenses_dir) if f.endswith(".npy")]
    nonlens_files = [f for f in os.listdir(nonlenses_dir) if f.endswith(".npy")]

    if len(lens_files) == 0 or len(nonlens_files) == 0:
        raise RuntimeError(
            "[Preflight] Dataset folders are empty or missing .npy files"
        )

    print("✓ Dataset folder structure looks valid")


def _check_indices(indices_path):
    with open(indices_path, "rb") as f:
        indices = pickle.load(f)

    required_splits = ["train", "val", "test"]
    required_classes = ["lenses", "nonlenses"]

    for split in required_splits:
        if split not in indices:
            raise KeyError(f"[Preflight] Missing split '{split}' in indices.pkl")

        for cls in required_classes:
            if cls not in indices[split]:
                raise KeyError(
                    f"[Preflight] Missing '{cls}' in indices['{split}']"
                )

            if len(indices[split][cls]) == 0:
                raise RuntimeError(
                    f"[Preflight] Empty split: indices['{split}']['{cls}']"
                )

    print("✓ indices.pkl structure is valid")


def _check_sample_image(args):
    data_path = args["input"]["data path"]
    expected_channels = args["input"]["channels"]

    sample_dir = os.path.join(data_path, "lenses")
    sample_file = next(
        f for f in os.listdir(sample_dir) if f.endswith(".npy")
    )

    sample_path = os.path.join(sample_dir, sample_file)
    sample = np.load(sample_path)
    sample_tensor = torch.from_numpy(sample)

    if sample_tensor.ndim == 2:
        sample_tensor = sample_tensor.unsqueeze(0)

    if sample_tensor.shape[0] != expected_channels:
        raise RuntimeError(
            f"[Preflight] Channel mismatch: expected {expected_channels}, "
            f"got {sample_tensor.shape[0]}"
        )

    if not torch.isfinite(sample_tensor).all():
        raise RuntimeError(
            "[Preflight] Sample image contains NaNs or infinite values"
        )

    print(
        f"✓ Sample image loaded successfully "
        f"(shape={tuple(sample_tensor.shape)})"
    )


def run_preflight_checks(args):
    print("\n[Preflight Check] Starting DeepLense SSL validation...\n")

    _check_paths(args)
    _check_dataset_structure(args["input"]["data path"])
    _check_indices(args["input"]["indices"])
    _check_sample_image(args)

    print("\n[Preflight Check] All checks passed successfully.")
