"""
Extract best hyperparameters from W&B sweeps and print them in hparams.py format.

Usage:
    python extract_best_hparams.py --project new_hparams --entity hussamaskar12-rostock
    python extract_best_hparams.py --project new_hparams --sweep_ids abc123 def456
    python extract_best_hparams.py --project new_hparams --methods DAAN ACON RAINCOAT

The script finds the best run (lowest trg_risk) per sweep, strips out the shared
train hparams (batch_size, num_epochs, etc.), and prints the result as a Python
dict ready to paste into configs/hparams.py.
"""

import argparse
import wandb

# Shared training hparams that belong in train_params, not alg_hparams
TRAIN_HPARAM_KEYS = {
    'num_epochs', 'batch_size', 'learning_rate', 'disc_lr',
    'weight_decay', 'step_size', 'optimizer', 'lr_decay',
}


def get_best_run(sweep, metric='trg_risk', goal='minimize'):
    """Return the best run in a sweep by the given metric."""
    runs = list(sweep.runs)
    if not runs:
        return None, None

    finished = [r for r in runs if r.state == 'finished']
    if not finished:
        finished = runs  # fall back to all runs if none finished cleanly

    def get_metric(run):
        val = run.summary.get(metric)
        return val if val is not None else float('inf') if goal == 'minimize' else float('-inf')

    best = min(finished, key=get_metric) if goal == 'minimize' else max(finished, key=get_metric)
    return best, get_metric(best)


def format_value(v):
    """Format a float for clean Python output."""
    if isinstance(v, float):
        # Use scientific notation for very small or very large values
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.2e}"
        return f"{v:.4g}"
    return repr(v)


def extract_alg_hparams(config: dict) -> dict:
    """Strip shared train keys, keep only algorithm-specific ones."""
    return {k: v for k, v in config.items()
            if k not in TRAIN_HPARAM_KEYS and not k.startswith('_')}


def extract_train_hparams(config: dict) -> dict:
    """Keep only shared training keys."""
    return {k: v for k, v in config.items()
            if k in TRAIN_HPARAM_KEYS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, help='W&B project name')
    parser.add_argument('--entity', default=None, help='W&B entity (username/team)')
    parser.add_argument('--metric', default='trg_risk', help='Metric to optimise for')
    parser.add_argument('--goal', default='minimize', choices=['minimize', 'maximize'])
    parser.add_argument('--sweep_ids', nargs='*', default=None,
                        help='Specific sweep IDs to extract (default: all in project)')
    parser.add_argument('--methods', nargs='*', default=None,
                        help='Filter by method name (matched against sweep name)')
    args = parser.parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity
    project_path = f"{entity}/{args.project}"

    if args.sweep_ids:
        sweeps = [api.sweep(f"{project_path}/{sid}") for sid in args.sweep_ids]
    else:
        project = api.project(args.project, entity=entity)
        sweeps = list(project.sweeps())

    if not sweeps:
        print("No sweeps found.")
        return

    results = {}  # method_name -> (best_run, metric_value, sweep)

    for sweep in sweeps:
        sweep_name = sweep.name or sweep.id
        method_name = sweep_name.split('_')[0] if '_' in sweep_name else sweep_name

        if args.methods and method_name not in args.methods:
            continue

        best_run, metric_val = get_best_run(sweep, args.metric, args.goal)
        if best_run is None:
            print(f"# Sweep {sweep_name}: no runs found, skipping")
            continue

        # Keep the best per method (lowest metric)
        if method_name not in results:
            results[method_name] = (best_run, metric_val, sweep_name)
        else:
            _, prev_val, _ = results[method_name]
            if (args.goal == 'minimize' and metric_val < prev_val) or \
               (args.goal == 'maximize' and metric_val > prev_val):
                results[method_name] = (best_run, metric_val, sweep_name)

    if not results:
        print("No matching sweeps/runs found.")
        return

    # Print hparams.py snippet
    print("\n# ============================================================")
    print("# Best hyperparameters extracted from W&B sweeps")
    print(f"# Project: {project_path}  |  Metric: {args.metric} ({args.goal})")
    print("# Paste into configs/hparams.py under self.alg_hparams = { ... }")
    print("# ============================================================\n")

    train_hparams_by_method = {}

    for method_name, (best_run, metric_val, sweep_name) in sorted(results.items()):
        cfg = dict(best_run.config)
        alg_hparams = extract_alg_hparams(cfg)
        train_hparams = extract_train_hparams(cfg)
        train_hparams_by_method[method_name] = train_hparams

        print(f"    # Sweep: {sweep_name}  |  {args.metric}: {metric_val:.4f}  |  run: {best_run.id}")
        print(f"    '{method_name}': {{", end="")
        items = list(alg_hparams.items())
        if not items:
            print("},")
        else:
            print()
            for k, v in items:
                print(f"        '{k}': {format_value(v)},")
            print("    },")
        print()

    # Also print the best shared train params per method for reference
    print("\n# ---- Suggested train_params (shared, from best runs) ----")
    for method_name, tp in sorted(train_hparams_by_method.items()):
        print(f"# {method_name}: ", end="")
        print(", ".join(f"{k}={format_value(v)}" for k, v in sorted(tp.items())))


if __name__ == '__main__':
    main()
