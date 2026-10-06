"""Sensor-only ADL forecasting with day-disjoint validation and test data."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def prepare_frame(frame, target):
    frame = frame.copy()
    if not {"Date", target}.issubset(frame):
        raise ValueError("Date and the requested target column are required.")
    if "House" not in frame:
        frame["House"] = "single_house"
    if frame[["Date", "House", target]].isna().any().any():
        raise ValueError("Missing timestamps, house identifiers, or target labels.")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if frame.duplicated(["House", "Date"]).any():
        raise ValueError("Duplicate house/timestamp rows; resolve before training.")
    frame["day"] = frame["Date"].dt.normalize()
    return frame.sort_values(["House", "Date"]).reset_index(drop=True)


def split_days(frame, test_days=2, val_days=1):
    days = sorted(frame["day"].unique())
    if min(test_days, val_days) < 1 or len(days) <= test_days + val_days:
        raise ValueError("Need positive validation/test day counts and earlier training days.")
    test = days[-test_days:]
    val = days[-test_days-val_days:-test_days]
    return (frame[~frame.day.isin(test + val)].copy(),
            frame[frame.day.isin(val)].copy(), frame[frame.day.isin(test)].copy())


def sensor_columns(frame):
    cols = [c for c in frame if c.endswith(("_motion", "_door", "_temp", "_light", "_hum", "_bin"))]
    if not cols:
        raise ValueError("No supported sensor columns found.")
    if not np.isfinite(frame[cols].to_numpy(dtype=float)).all():
        raise ValueError("Sensor values must be finite; resolve missing readings first.")
    return cols


def features(frame, cols):
    """Causal room-specific summaries; restart at each house/day boundary."""
    pieces = []
    for _, group in frame.groupby(["House", "day"], sort=False):
        x = group[cols].astype(float).copy()
        seconds = group.Date.diff().dt.total_seconds()
        minutes = group.Date.dt.hour * 60 + group.Date.dt.minute
        x["time_sin"] = np.sin(2 * np.pi * minutes / 1440)
        x["time_cos"] = np.cos(2 * np.pi * minutes / 1440)
        for col in cols:
            change = group[col].diff().fillna(0)
            if col.endswith(("_motion", "_door", "_bin")):
                x[col + "_count10"] = group[col].rolling(10, min_periods=1).sum()
                x[col + "_changed"] = change.ne(0).astype(float)
            else:
                x[col + "_mean10"] = group[col].rolling(10, min_periods=1).mean()
                x[col + "_std10"] = group[col].rolling(10, min_periods=1).std().fillna(0)
                x[col + "_rate_per_second"] = (change / seconds).fillna(0)
        pieces.append(x)
    return pd.concat(pieces).reindex(frame.index)


def sequences(frame, x, y, length, max_gap_seconds=300):
    """Predict the NEXT row; never bridge houses, days, or large recording gaps."""
    windows, labels = [], []
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    for _, group in frame.groupby(["House", "day"], sort=False):
        gaps = group.Date.diff().dt.total_seconds().fillna(0).gt(max_gap_seconds).cumsum()
        for _, segment in group.groupby(gaps):
            idx = positions.loc[segment.index].to_numpy()
            for end in range(length, len(idx)):
                windows.append(x[idx[end-length:end]])
                labels.append(y[idx[end]])
    if not windows:
        raise ValueError("No complete windows: reduce sequence length or check sampling gaps.")
    return TensorDataset(torch.tensor(np.stack(windows), dtype=torch.float32),
                         torch.tensor(labels, dtype=torch.long))


class ADLModel(nn.Module):
    def __init__(self, input_dim, classes, kind="transformer", hidden=32, layers=2):
        super().__init__()
        self.kind = kind
        if kind == "transformer":
            self.project = nn.Linear(input_dim, hidden)
            position = torch.arange(5000).unsqueeze(1)
            div = torch.exp(torch.arange(0, hidden, 2) * (-math.log(10000) / hidden))
            pe = torch.zeros(5000, hidden)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div), torch.cos(position * div)
            self.register_buffer("pe", pe)
            layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 4, dropout=0.2,
                                                activation="gelu", batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers)
        else:
            self.encoder = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                                   dropout=0.2 if layers > 1 else 0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, classes))

    def forward(self, x):
        if self.kind == "transformer":
            # All input rows precede the prediction target, so no future inputs are visible.
            x = self.encoder(self.project(x) + self.pe[:x.size(1)])
        else:
            x, _ = self.encoder(x)
        return self.head(x[:, -1])


def train(model, train_loader, val_loader, epochs, patience, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        losses = []
        for training, loader in [(True, train_loader), (False, val_loader)]:
            model.train(training)
            total, count = 0.0, 0
            with torch.set_grad_enabled(training):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    loss = criterion(model(x), y)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    total += loss.item() * len(y)
                    count += len(y)
            losses.append(total / count)
        history.append({"epoch": epoch + 1, "train_loss": losses[0], "val_loss": losses[1]})
        if losses[1] < best:
            best, stale = losses[1], 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(state)
    return history


def evaluate(model, loader, classes, device):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probabilities.append(model(x.to(device)).softmax(-1).cpu().numpy())
            labels.extend(y.tolist())
    prob = np.concatenate(probabilities)
    y = np.asarray(labels)
    pred = prob.argmax(1)
    indices = np.arange(len(classes))
    binary = (y[:, None] == indices).astype(int)
    per_auc = [roc_auc_score(binary[:, i], prob[:, i])
               if len(np.unique(binary[:, i])) == 2 else None for i in indices]
    return {"accuracy": accuracy_score(y, pred),
            "f1_micro": f1_score(y, pred, average="micro"),
            "f1_macro": f1_score(y, pred, labels=indices, average="macro", zero_division=0),
            "auc_micro_ovr": roc_auc_score(binary.ravel(), prob.ravel()),
            "auc_macro_ovr": float(np.mean(per_auc)) if all(a is not None for a in per_auc) else None,
            "per_class_auc": dict(zip(classes, per_auc)),
            "classification_report": classification_report(y, pred, labels=indices,
                target_names=classes, output_dict=True, zero_division=0)}


def main(kind="transformer"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default="P", choices=["P", "C"])
    parser.add_argument("--seq-len", type=int, default=30, help="Rows, not necessarily minutes")
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--val-days", type=int, default=1)
    parser.add_argument("--max-gap-seconds", type=float, default=300)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.seq_len, args.epochs, args.patience, args.batch_size, args.max_gap_seconds) <= 0 or args.seq_len > 5000:
        parser.error("Require positive settings and sequence length <= 5000.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new or empty output directory.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    frame = prepare_frame(pd.read_csv(args.csv), args.target)
    cols = sensor_columns(frame)
    parts = split_days(frame, args.test_days, args.val_days)
    enc = LabelEncoder().fit(parts[0][args.target].astype(str))
    if len(enc.classes_) < 2:
        raise ValueError("Training requires at least two target classes.")
    engineered = [features(part, cols) for part in parts]
    scaler = StandardScaler().fit(engineered[0])
    datasets = []
    for part, feat in zip(parts, engineered):
        # Unseen classes raise an error instead of silently removing test observations.
        y = enc.transform(part[args.target].astype(str))
        datasets.append(sequences(part, scaler.transform(feat), y, args.seq_len, args.max_gap_seconds))
    loaders = [DataLoader(ds, args.batch_size, shuffle=i == 0) for i, ds in enumerate(datasets)]
    device = torch.device(args.device)
    model = ADLModel(engineered[0].shape[1], len(enc.classes_), kind).to(device)
    start = time.perf_counter()
    history = train(model, *loaders[:2], args.epochs, args.patience, device)
    metrics = evaluate(model, loaders[2], enc.classes_.tolist(), device)
    metrics.update(model=kind, epochs=len(history), parameters=sum(p.numel() for p in model.parameters()),
                   elapsed_seconds=time.perf_counter() - start)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    # Artifacts stay local; checkpoints trained on governed data must not be published.
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "classes": enc.classes_.tolist(), "feature_columns": list(engineered[0]),
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                "kind": kind, "hidden": 32, "layers": 2, "seq_len": args.seq_len},
               args.output_dir / "model.pt")
    print(json.dumps(metrics, indent=2, allow_nan=False))
