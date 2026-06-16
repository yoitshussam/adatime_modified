"""Pull the best trial per method from an MLflow sweep and write a JSON file
shaped like `alg_hparams` in configs/hparams.py — ready to paste into
`self.alg_hparams = {...}`:

    {
      "DANN": { "learning_rate": 0.0005, "src_cls_loss_wt": 3.1, ... },
      "DSAN": { "learning_rate": 0.001,  "mmd_wt": 4.2,          ... },
      ...
    }

By default the best trial per method is the one with the lowest avg_trg_risk
(matching main_sweep.py's default --metric_to_minimize). Use --metric / --goal
to optimize a different logged metric (e.g. avg_f1_score, maximize).

Run:
    python extract_best_hparams.py --exp_name sweep_EXP1
    python extract_best_hparams.py --exp_name sweep_EXP1 --metric avg_f1_score --goal maximize
"""
import argparse
import json
import math
import os
import re
import sys

import mlflow
from mlflow.tracking import MlflowClient

METHODS = ['Deep_Coral', 'DDC', 'MMDA', 'DANN', 'CDAN', 'DIRT', 'DSAN', 'HoMM',
           'CoDATS', 'AdvSKM', 'SASA', 'CoTMix', 'SWL_Adapt', 'uDAR', 'DAAN',
           'ACON', 'RAINCOAT', 'SSSS_TSA', 'CLUDA']

# Swept params that must be ints (MLflow stores all params as strings).
INT_KEYS = {"batch_size", "num_epochs", "step_size", "temporal_shift",
            "WA_N_hid", "acon_disc_hid_dim", "tau_temp"}

# Run metadata logged alongside hyperparameters — filtered out of the JSON.
CONTEXT_KEYS = {"source_dataset", "target_dataset", "backbone", "da_method"}

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "best_hparams.json")


def coerce(key, value):
    """MLflow stores params as strings — turn them back into int/float."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if key in INT_KEYS:
        return int(round(f))
    return f


def parse_method(run_name):
    if not run_name:
        return None
    m = re.match(r"^([A-Za-z_]+)_trial_\d+$", run_name)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True,
                    help="The --exp_name that was passed to main_sweep.py")
    ap.add_argument("--source_dataset", default="RealWorld_Male")
    ap.add_argument("--target_dataset", default="RealWorld_Female")
    ap.add_argument("--metric", default="avg_trg_risk",
                    help="Logged summary metric to select the best trial by")
    ap.add_argument("--goal", default="minimize", choices=["minimize", "maximize"])
    args = ap.parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    client = MlflowClient()

    experiment_name = (f"sweep_{args.source_dataset}_to_"
                       f"{args.target_dataset}_{args.exp_name}")
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        sys.exit(f"Experiment not found: {experiment_name!r}")

    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=10000)

    maximize = args.goal == "maximize"
    best = {}  # method -> (score, run)
    for r in runs:
        method = parse_method(r.data.tags.get("mlflow.runName"))
        if method not in METHODS:
            continue
        score = r.data.metrics.get(args.metric)
        if score is None or not math.isfinite(score):
            continue
        if method not in best:
            best[method] = (score, r)
        else:
            cur = best[method][0]
            if (score > cur) if maximize else (score < cur):
                best[method] = (score, r)

    if not best:
        sys.exit(f"No completed runs with metric {args.metric!r} in {experiment_name!r}")

    payload = {
        method: {k: coerce(k, v)
                 for k, v in best[method][1].data.params.items()
                 if k not in CONTEXT_KEYS}
        for method in METHODS if method in best
    }

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=4, sort_keys=True)
    print(f"Wrote {len(payload)} methods -> {OUT_PATH} "
          f"(best by {args.goal} {args.metric})")


if __name__ == "__main__":
    main()
