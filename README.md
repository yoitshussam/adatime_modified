# AdaTime — Closed-Set Domain Adaptation for Time-Series HAR

A benchmark for **closed-set unsupervised domain adaptation (UDA)** on time-series
Human Activity Recognition (HAR). Nineteen adaptation methods — spanning global
(marginal) alignment and local (class-conditional) alignment families — are run
under a single training/evaluation protocol across three HAR datasets, with
Weights & Biases hyper-parameter sweeps, MLflow experiment tracking, and a
notebook-based analysis stage that produces confusion matrices, per-class
metrics, and global-vs-local comparison figures.

## 1. Requirements

The codebase targets recent PyTorch; matching the versions below is the safest
bet, anything reasonably close should also work.

| Package        | Notes                                  |
|----------------|----------------------------------------|
| python         | 3.9+                                   |
| torch          | CUDA build recommended (`--device cuda`) |
| numpy, pandas, scipy, scikit-learn |                    |
| matplotlib, seaborn | analysis notebook figures         |
| mlflow         | experiment tracking (training)         |
| wandb          | hyper-parameter sweeps                 |
| optuna         | sweep search backend                   |
| tqdm           |                                        |

### Start an MLflow server first

`main.py` logs to MLflow over HTTP at `http://127.0.0.1:5001`. Launch the
tracking server **before** training:

```bash
mlflow server --host 127.0.0.1 --port 5001
```

To use a different port, change it both on the server command line and in the
`mlflow.set_tracking_uri(...)` call at the top of `main.py`.

Sweeps (`main_sweep.py`) and `extract_best_hparams.py` use **Weights & Biases**
— run `wandb login` once before sweeping.

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
(`sweep_train_hparams`, `sweep_alg_hparams`). `main_sweep.py` runs W&B/Optuna
trials; `extract_best_hparams.py` pulls the best trial per method from the W&B
project.

```bash
# Sweep one method:
python main_sweep.py \
    --source_dataset RealWorld_Male \
    --target_dataset RealWorld_Female \
    --da_method DAAN \
    --num_runs 3 \
    --num_sweeps 50 \
    --hp_search_strategy bayes \
    --metric_to_minimize trg_risk \
    --exp_name sweep_EXP1

# Extract best hparams for chosen methods from the W&B project:
python extract_best_hparams.py \
    --project new_hparams \
    --entity <your-wandb-entity> \
    --methods DAAN ACON RAINCOAT \
    --metric trg_risk --goal minimize
```

The extracted values are a reference artifact — copy them into the matching
`alg_hparams[<method>]` entry in `configs/hparams.py` before re-running
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
  sweep_params.py           — W&B/Optuna sweep search spaces
dataloader/dataloader.py    — windowed-data loaders (data_generator)
models/models.py            — CNN / HAR_CNN backbones + classifiers
data_pre_processing.py      — raw HAR → sliding-window pipeline
load_data.py                — labelled/unlabelled split loader
trainers/
  abstract_trainer.py       — shared train/eval/checkpoint logic
  train.py                  — Trainer.fit() (single config, MLflow logging)
  sweep.py                  — Trainer.sweep() (W&B hyper-parameter search)
main.py                     — single-config / ALL-methods trainer
main_sweep.py               — hyper-parameter sweep entry point
extract_best_hparams.py     — pull best sweep hparams from W&B
new_fixed.ipynb             — analysis: confusion matrices + comparison figures
run_experiments*.sh         — full 6-pair training wrappers
experiments_logs/           — per-run checkpoints, result CSVs, pics/, confusion_matrices/
```
