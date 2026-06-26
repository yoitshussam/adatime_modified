# AdaTime — Closed-Set Domain Adaptation for Time-Series HAR

A benchmark for **closed-set unsupervised domain adaptation (UDA)** on time-series
Human Activity Recognition (HAR). Nineteen adaptation methods — spanning global
(marginal) alignment and local (class-conditional) alignment families — are run
under a single training/evaluation protocol across three HAR datasets, with
Optuna hyper-parameter sweeps, MLflow experiment tracking, and a
notebook-based analysis stage that produces confusion matrices, per-class
metrics, and global-vs-local comparison figures.

## 1. Requirements

The codebase is developed against the versions below; matching them exactly is
the safest bet, anything reasonably close should also work.

| Package        | Version      |
|----------------|--------------|
| python         | 3.9.25       |
| torch          | 2.7.1+cu118  |
| torchvision    | 0.22.1+cu118 |
| numpy          | 2.0.2        |
| pandas         | 2.3.3        |
| scipy          | 1.13.1       |
| scikit-learn   | 1.6.1        |
| matplotlib     | 3.9.4        |
| seaborn        | 0.13.2       |
| mlflow         | 3.1.4        |
| optuna         | 4.8.0        |
| tqdm           | 4.67.1       |

### Start an MLflow server first

`main.py`, `main_sweep.py`, and `extract_best_hparams.py` all log to / read from
MLflow over HTTP at `http://127.0.0.1:5001`. Launch the tracking server
**before** training or sweeping:

```bash
mlflow server --host 127.0.0.1 --port 5001
```

To use a different port, change it both on the server command line and in the
`mlflow.set_tracking_uri(...)` calls at the top of `main.py` / `main_sweep.py` /
`extract_best_hparams.py`.

Hyper-parameter search uses **Optuna** (no external login or account required) —
trials are written straight to MLflow. The old Weights & Biases sweep scripts are
kept for reference under `wandbarchive/` (git-ignored) and are no longer used.

## 2. Datasets

Three public HAR datasets are used, all resampled and windowed to a common
format (`sequence_len = 150`, `input_channels = 18`):

| Dataset    | `--source_dataset` / `--target_dataset` value |
|------------|-----------------------------------------------|
| RealWorld  | `RealWorld`                                   |
| Pamap2     | `Pamap2`                                       |
| MHEALTH    | `mhealth`                                      |

The five shared classes — **sitting / standing / lying / running / walking** —
form the closed-set label space (`num_classes = 5`), defined in
`configs/data_model_configs.py` (`ALL` class).

### Preprocessing

The raw → windowed pipeline lives in `data_pre_processing.py` (sliding-window
segmentation, activity remapping) and `load_data.py`
(`load_labelled_and_unlabelled`, train/val/test splitting). The dataloader
expects one **`<dataset>_processed.pkl`** per dataset under `--data_path`
(default `../dataset`), e.g. `../dataset/RealWorld_processed.pkl`. At load time
`dataloader/data_generator` reads the labelled split for the source and the
unlabelled split for the target.

## 3. Algorithms

All algorithms live in `algorithms/algorithms.py`, each a subclass of
`Algorithm`, resolved by name through `get_algorithm_class()`. Selectable via
`--da_method <NAME>` or `--da_method ALL` to run the whole list.

Two baselines: **NO_ADAPT** (source-only) and **TARGET_ONLY** (oracle).
The nineteen adaptation methods, organized by alignment scope × mechanism:

### Global alignment (marginal `P(x)`)

| Mechanism            | Methods                          |
|----------------------|----------------------------------|
| Discrepancy-based    | DDC (MMD), Deep_Coral (CORAL), HoMM (higher moments), RAINCOAT (Sinkhorn/OT + reconstruction), SSSS_TSA (Sinkhorn/OT) |
| Adversarial          | DANN, CoDATS, AdvSKM             |

### Local alignment (global + class-conditional)

| Mechanism                 | Methods                                  |
|---------------------------|------------------------------------------|
| Discrepancy-based         | DSAN (LMMD), uDAR (class-wise MMD), MMDA, SASA (structural MMD) |
| Adversarial               | CDAN, DAAN, ACON                         |
| Decision-boundary / posterior / self-training | DIRT (VAT+entropy), MCD (classifier discrepancy), SWL_Adapt (pseudo-labels + reweighting) |
| Contrastive / self-supervised | CLUDA, CoTMix                        |

> Notes: **MMDA** is marginal MMD+CORAL with a conditional-entropy regularizer
> (sits between global and local); **SASA** aligns sparse associative structure
> with no explicit label use. See `algorithms/algorithms.py` for the exact loss
> composition of each method.

Backbones live in `models/models.py` (`get_backbone_class`); default is `CNN`,
with `HAR_CNN` also available via `--backbone HARCNN`.

## 4. Training procedure

The pipeline is **preprocess → (sweep → extract) → train → analyze**.

### 4.1 Train a single configuration

`main.py` runs one (source, target, method) configuration for `--num_runs`
seeds and logs to MLflow.

```bash
# One method on one pair:
python main.py \
    --source_dataset RealWorld \
    --target_dataset Pamap2 \
    --da_method DSAN \
    --backbone CNN \
    --num_runs 5 \
    --exp_name EXP3

# Or run every method on one pair:
python main.py --source_dataset RealWorld --target_dataset Pamap2 \
               --da_method ALL --num_runs 5 --exp_name EXP3
```

Hyper-parameters come from `configs/hparams.py` (`get_hparams_class`), which
holds `train_params` (epochs, etc.) and `alg_hparams[<method>]` (per-method
learning rate, loss weights, batch size).

Outputs land under
`experiments_logs/<source> to <target>_<exp_name>/<method>_<exp_name>/`:
- `<source>_to_<target>_run_<n>/checkpoint.pt` + training log, per seed
- `trg_results.csv`, `src_results.csv`, `risks.csv` (per-method, with mean/std rows)

The example shell wrappers run the full 6-pair grid:

```bash
bash run_experiments_exp1.sh   # closed_set, 5 seeds, all pairs
bash run_experiments.sh        # EXP_2, 10 seeds, all pairs
```

### 4.2 Hyper-parameter sweep + extraction

Sweep search spaces are defined in `configs/sweep_params.py`
(`sweep_train_hparams`, `sweep_alg_hparams`). `main_sweep.py` runs **Optuna**
trials and logs every trial to MLflow; `extract_best_hparams.py` reads that
MLflow experiment back and pulls the best trial per method.

The sweep split is `RealWorld_Male` → `RealWorld_Female` (held-out, separate from
the evaluation pairs). Each trial samples hyper-parameters from
`configs/sweep_params.py`, trains `--num_runs` seeds, and logs the run-averaged
metrics (`avg_acc`, `avg_f1_score`, `avg_auroc`, `avg_src_risk`, `avg_trg_risk`).

`--hp_search_strategy` selects the Optuna sampler: `bayes` (TPE, default),
`random`, or `grid`. `--metric_to_minimize` chooses the objective — `trg_risk`
(default) or `src_risk` are minimized; `f1_score`, `acc`, `auroc` are maximized
(negated internally).

```bash
# Sweep one method (50 Optuna trials, 3 seeds each):
python main_sweep.py \
    --source_dataset RealWorld_Male \
    --target_dataset RealWorld_Female \
    --da_method DAAN \
    --num_runs 3 \
    --num_sweeps 50 \
    --hp_search_strategy bayes \
    --metric_to_minimize trg_risk \
    --exp_name sweep_EXP1

# Or sweep every method in one go:
python main_sweep.py --da_method ALL --num_runs 3 --num_sweeps 50 \
                     --exp_name sweep_EXP1

# Extract the best trial per method from the MLflow sweep experiment:
python extract_best_hparams.py \
    --exp_name sweep_EXP1 \
    --source_dataset RealWorld_Male \
    --target_dataset RealWorld_Female \
    --metric avg_trg_risk --goal minimize
```

`main_sweep.py` writes each trial to the MLflow experiment
`sweep_<source>_to_<target>_<exp_name>`, with run names `<method>_trial_<n>`.
`extract_best_hparams.py` reads that experiment, picks the best trial per method
by `--metric` / `--goal` (default: lowest `avg_trg_risk`), and writes
`best_hparams.json`. Those values are a reference artifact — copy them into the
matching `alg_hparams[<method>]` entry in `configs/hparams.py` before re-running
`main.py`.

### 4.3 Analysis: confusion matrices & comparison figures

The analysis stage is the notebook **`new_fixed.ipynb`** (most current; older
variants `new.ipynb` / `new2.ipynb` are kept for reference). It:

1. Loads each method's `checkpoint.pt`, reconstructs the model, runs the target
   test set, and writes per-run confusion matrices to
   `experiments_logs/confusion_matrices/<EXP>/<method>/*_cm.txt`.
2. Aggregates those into a grid of confusion-matrix heatmaps, a
   misclassification summary, and per-class F1 / group-metric comparison bars —
   sorted and color-coded by the **Global / Local** grouping (see
   `get_algo_sort_key`).

Figures are written under `experiments_logs/pics/<EXP>/` (grids, metrics
analysis, misclassification summaries, and `individual_cms/<scenario>/`).

## 5. Repository layout

```
algorithms/algorithms.py    — every DA algorithm (Algorithm subclasses)
configs/
  data_model_configs.py     — dataset + backbone config (ALL class)
  hparams.py                — train_params + per-method alg_hparams
  sweep_params.py           — Optuna sweep search spaces
dataloader/dataloader.py    — windowed-data loaders (data_generator)
models/models.py            — CNN / HAR_CNN backbones + classifiers
data_pre_processing.py      — raw HAR → sliding-window pipeline
load_data.py                — labelled/unlabelled split loader
trainers/
  abstract_trainer.py       — shared train/eval/checkpoint logic
  train.py                  — Trainer.fit() (single config, MLflow logging)
  sweep.py                  — Trainer.sweep() (Optuna search, MLflow logging)
main.py                     — single-config / ALL-methods trainer
main_sweep.py               — Optuna hyper-parameter sweep entry point
extract_best_hparams.py     — pull best sweep hparams from MLflow
wandbarchive/               — old W&B sweep scripts (git-ignored, unused)
new_fixed.ipynb             — analysis: confusion matrices + comparison figures
run_experiments*.sh         — full 6-pair training wrappers
experiments_logs/           — per-run checkpoints, result CSVs, pics/, confusion_matrices/
```
