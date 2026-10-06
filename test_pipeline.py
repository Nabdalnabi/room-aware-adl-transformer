import numpy as np
import pytest
import torch
from adl_pipeline import ADLModel, features, prepare_frame, sensor_columns, sequences, split_days
from make_synthetic_demo import make_frame


def test_split_and_features():
    frame = prepare_frame(make_frame(), "P")
    train, val, test = split_days(frame)
    assert max(train.day) < min(val.day) < min(test.day)
    cols = sensor_columns(frame)
    x = features(train, cols)
    assert "P" not in x and "House" not in x
    changed = train.copy()
    changed.loc[changed.index[-1], "kitchen_temp"] += 1000
    assert x.iloc[:-1].equals(features(changed, cols).iloc[:-1])


def test_windows_and_models():
    frame = prepare_frame(make_frame(), "P")
    x = features(frame, sensor_columns(frame)).to_numpy()
    ds = sequences(frame, x, np.zeros(len(frame)), 10)
    assert len(ds) == 6 * 90
    for kind in ["transformer", "lstm"]:
        model = ADLModel(x.shape[1], 2, kind)
        out = model(torch.stack([ds[0][0], ds[1][0]]))
        assert out.shape == (2, 2)
        assert torch.isfinite(out).all()


def test_duplicates_fail():
    frame = make_frame()
    frame.loc[1, "Date"] = frame.loc[0, "Date"]
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_frame(frame, "P")
