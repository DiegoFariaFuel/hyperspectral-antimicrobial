from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import fit_band_normalization, load_npz, normalize_cubes
from .models import build_model, model_input


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train(args: argparse.Namespace) -> None:
    train_set = load_npz(args.train)
    validation_set = load_npz(args.validation, train_set.class_names)
    if train_set.cubes.shape[1:] != validation_set.cubes.shape[1:]:
        raise ValueError("training and validation cubes must have the same H, W, and band dimensions")

    mean, std = fit_band_normalization(train_set.cubes)
    x_train = model_input(torch.from_numpy(normalize_cubes(train_set.cubes, mean, std)), args.model)
    x_val = model_input(torch.from_numpy(normalize_cubes(validation_set.cubes, mean, std)), args.model)
    y_train = torch.from_numpy(train_set.labels)
    y_val = torch.from_numpy(validation_set.labels)

    device = _device(args.device)
    model = build_model(args.model, len(train_set.class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                losses.append(criterion(model(batch_x.to(device)), batch_y.to(device)).item())
        val_loss = float(np.mean(losses))
        print(f"epoch={epoch:03d} validation_loss={val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a model checkpoint")
    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model,
            "state_dict": best_state,
            "class_names": list(train_set.class_names),
            "band_count": int(train_set.cubes.shape[-1]),
            "height": int(train_set.cubes.shape[1]),
            "width": int(train_set.cubes.shape[2]),
            "band_mean": mean.tolist(),
            "band_std": std.tolist(),
        },
        output,
    )
    print(f"saved={output}")


def evaluate(args: argparse.Namespace) -> None:
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    class_names = tuple(checkpoint["class_names"])
    data = load_npz(args.data, class_names)
    if data.cubes.shape[1:] != (checkpoint["height"], checkpoint["width"], checkpoint["band_count"]):
        raise ValueError("test cubes do not match the dimensions used to train the model")

    mean = np.asarray(checkpoint["band_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["band_std"], dtype=np.float32)
    x = model_input(torch.from_numpy(normalize_cubes(data.cubes, mean, std)), checkpoint["model_name"])
    model = build_model(checkpoint["model_name"], len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    with torch.no_grad():
        logits = model(x.to(device)).cpu()
    predicted = logits.argmax(dim=1).numpy()

    report = {
        "sample_count": int(len(predicted)),
        "classes": list(class_names),
        "balanced_accuracy": float(balanced_accuracy_score(data.labels, predicted)),
        "macro_f1": float(f1_score(data.labels, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(data.labels, predicted, labels=np.arange(len(class_names))).tolist(),
        "classification_report": classification_report(
            data.labels, predicted, labels=np.arange(len(class_names)), target_names=class_names,
            output_dict=True, zero_division=0,
        ),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="hsi-amr")
    commands = parser.add_subparsers(required=True)
    train_parser = commands.add_parser("train", help="train a 1D spectral or 3D cube CNN")
    train_parser.add_argument("--train", required=True)
    train_parser.add_argument("--validation", required=True)
    train_parser.add_argument("--model", choices=("spectral", "cube3d"), default="spectral")
    train_parser.add_argument("--checkpoint", required=True)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--patience", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--device", default="auto")
    train_parser.set_defaults(func=train)

    eval_parser = commands.add_parser("evaluate", help="evaluate a checkpoint on held-out data")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--data", required=True)
    eval_parser.add_argument("--output")
    eval_parser.add_argument("--device", default="auto")
    eval_parser.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
