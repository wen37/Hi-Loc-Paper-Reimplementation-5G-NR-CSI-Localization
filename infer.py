import argparse
import torch

from utils.runtime import (
    prepare_runtime,
    build_dataset,
    build_model,
    create_data_splits,
    load_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="InF_DH")
    parser.add_argument("--sync_err", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = prepare_runtime()
    device = config["train"]["device"]
    print(f"Using device: {device}")

    dataset = build_dataset(config, scene=args.scene, sync_err=args.sync_err)
    train_dataset, val_dataset, test_dataset = create_data_splits(dataset, config)
    split_map = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    target_dataset = split_map[args.split]

    if len(target_dataset) == 0:
        raise ValueError(f"Split '{args.split}' is empty.")
    if args.index < 0 or args.index >= len(target_dataset):
        raise IndexError(
            f"Index {args.index} out of range for split '{args.split}' "
            f"with size {len(target_dataset)}."
        )

    model = build_model(config).to(device)
    checkpoint = load_checkpoint(
        model, config["train"]["checkpoint_path"], device
    )
    model.eval()

    x, y = target_dataset[args.index]
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()

    abs_error = torch.abs(pred - y)
    distance_error = torch.norm(pred - y).item()

    print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")
    print(f"Scene: {args.scene}, sync_err: {args.sync_err}, split: {args.split}, index: {args.index}")
    print(f"Ground Truth (x, y): {y.tolist()}")
    print(f"Prediction   (x, y): {pred.tolist()}")
    print(f"Absolute Error     : {abs_error.tolist()}")
    print(f"Distance Error     : {distance_error:.4f}")


if __name__ == "__main__":
    main()
