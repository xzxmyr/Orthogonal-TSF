"""Load one frozen R6 model and report predictions; never fit a model."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
from model import StateHistory
from run import dgp


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", default="s1429_regular_p1_j1_ar1")
    parser.add_argument("--variant", default="likelihood_clean_state")
    args = parser.parse_args()
    rows = json.loads((root / "artifacts/metrics.json").read_text())
    row = next((r for r in rows if r["id"] == args.configuration and
                r["variant"] == args.variant), None)
    if row is None:
        parser.error("configuration/variant is absent from the frozen release")
    checkpoint = root / "artifacts" / args.configuration / (args.variant + ".npz")
    m = StateHistory.load(checkpoint)
    t, y, x, _ = dgp(row["seed"], row["period"], row["interaction"],
                     row["design"], row["ar"])
    idx = np.arange(3601, 4801)
    out = m.predict(t, y, x, idx)
    f0 = m.coef[0] + m.coef[1] * t[idx] / 3600 + m.coef[2] * y[idx - 1]
    predicted = f0 + out["prediction_total"]
    print(json.dumps({"configuration": args.configuration, "variant": args.variant,
                      "no_training": True, "rolling_predictions": len(idx),
                      "prediction_mse": float(np.mean((y[idx] - predicted) ** 2)),
                      "first_prediction": float(predicted[0]),
                      "first_residual_contributions": {k: float(out[k][0]) for k in
                      ("bias", "T", "M", "J", "H", "H_state", "H_remainder",
                       "H_calibration")}}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
