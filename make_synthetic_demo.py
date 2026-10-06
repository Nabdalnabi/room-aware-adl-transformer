"""Create fictional sensor observations solely for software smoke tests."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make_frame():
    rng = np.random.default_rng(42)
    rows = []
    for day in range(6):
        for minute in range(100):
            active = (minute // 10) % 2
            rows.append({"Date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day, minutes=minute),
                         "House": "synthetic_house", "kitchen_motion": active,
                         "bedroom_motion": 1-active, "kitchen_temp": 20 + rng.normal(),
                         "kitchen_light": 100 + 100*active + rng.normal(),
                         "door1_door": int(minute % 20 == 0),
                         "P": "synthetic_A" if active else "synthetic_B"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_frame().to_csv(args.output, index=False)
