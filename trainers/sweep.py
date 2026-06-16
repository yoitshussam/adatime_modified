import sys
sys.path.append('../')
import torch
import os
import gc
import time
import math
import optuna
import mlflow
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections

from configs.sweep_params import sweep_alg_hparams, sweep_train_hparams
from utils import fix_randomness, starting_logs, AverageMeter
from trainers.abstract_trainer import AbstractTrainer
from dataloader.dataloader import data_generator_pt

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def _safe_metrics(d):
    """Drop NaN/Inf values — MLflow cannot serialize them."""
    return {k: float(v) for k, v in d.items()
            if v is not None and math.isfinite(float(v))}


def sample_hparam(trial, name, spec):
    """Sample one hyperparameter from an Optuna trial given a sweep_params spec.

    Specs are kept in the existing wandb dict format so the tuned ranges in
    configs/sweep_params.py are preserved verbatim:
        {'values': [...]}                                  -> categorical
        {'distribution': 'uniform', 'min': a, 'max': b}    -> float
        {'distribution': 'log_uniform_values'/'log_uniform', 'min', 'max'} -> log float
        {'distribution': 'int_uniform', 'min', 'max'}      -> int
    """
    if 'values' in spec:
        return trial.suggest_categorical(name, spec['values'])

    dist = spec.get('distribution', 'uniform')
    lo, hi = spec['min'], spec['max']
    if dist in ('log_uniform_values', 'log_uniform'):
        return trial.suggest_float(name, lo, hi, log=True)
    if dist in ('int_uniform', 'q_uniform'):
        return trial.suggest_int(name, int(lo), int(hi))
    return trial.suggest_float(name, lo, hi)


class Trainer(AbstractTrainer):
    """Sweep trainer: Optuna for hyperparameter search, MLflow for logging."""

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # sweep parameters
        self.num_sweeps = args.num_sweeps
        self.hp_search_strategy = args.hp_search_strategy   # random, grid, bayes
        self.metric_to_minimize = args.metric_to_minimize   # trg_risk, src_risk, f1_score, acc, auroc

        self.results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        self.risks_columns = ["scenario", "run", "src_risk", "trg_risk"]

        # sweeps log to a flat directory (per-trial CSVs + checkpoints)
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir)
        os.makedirs(self.exp_log_dir, exist_ok=True)

    def load_data(self):
        """Sweep datasets are pre-split .pt files (train/val/test)."""
        self.src_train_dl = data_generator_pt(self.source_data_path, self.dataset_configs, self.hparams, 'source', "train")
        self.src_test_dl  = data_generator_pt(self.source_data_path, self.dataset_configs, self.hparams, 'source', "test")
        self.src_val_dl   = data_generator_pt(self.source_data_path, self.dataset_configs, self.hparams, 'source', "val")

        self.trg_train_dl = data_generator_pt(self.target_data_path, self.dataset_configs, self.hparams, 'target', "train")
        self.trg_test_dl  = data_generator_pt(self.target_data_path, self.dataset_configs, self.hparams, 'target', "test")

    def sweep(self):
        """Run an Optuna hyperparameter sweep with MLflow logging."""
        if self.hp_search_strategy == 'random':
            sampler = optuna.samplers.RandomSampler(seed=42)
        elif self.hp_search_strategy == 'grid':
            sampler = optuna.samplers.GridSampler(self._build_grid_search_space())
        else:  # 'bayes' (default)
            sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=3)

        study_name = f"{self.da_method}_{self.backbone}_{self.source_dataset}_to_{self.target_dataset}"
        study = optuna.create_study(study_name=study_name, direction='minimize', sampler=sampler)
        # catch=(Exception,) so a NaN/assertion in one trial doesn't sink the rest.
        study.optimize(self._objective, n_trials=self.num_sweeps, catch=(Exception,))

        try:
            best = study.best_trial
            print(f"\n===== Best Trial ({self.da_method}) =====")
            print(f"  Value ({self.metric_to_minimize}): {best.value:.4f}")
            print(f"  Params: {best.params}")
        except ValueError as e:
            print(f"\n===== No successful trials for {self.da_method}: {e} =====")
        return study

    def _build_grid_search_space(self):
        """Build a grid search space dict for Optuna's GridSampler."""
        space = {}
        merged = {**sweep_train_hparams, **sweep_alg_hparams.get(self.da_method, {})}
        for name, spec in merged.items():
            if 'values' in spec:
                space[name] = spec['values']
            else:
                lo, hi = spec['min'], spec['max']
                dist = spec.get('distribution', 'uniform')
                if dist in ('log_uniform_values', 'log_uniform'):
                    space[name] = np.logspace(np.log10(lo), np.log10(hi), 5).tolist()
                elif dist in ('int_uniform', 'q_uniform'):
                    space[name] = list(range(int(lo), int(hi) + 1))
                else:
                    space[name] = np.linspace(lo, hi, 5).tolist()
        return space

    def _sample_hparams(self, trial):
        """Sample all hyperparameters for a trial (train + algorithm-specific)."""
        hparams = {}
        for name, spec in sweep_train_hparams.items():
            hparams[name] = sample_hparam(trial, name, spec)
        for name, spec in sweep_alg_hparams.get(self.da_method, {}).items():
            hparams[name] = sample_hparam(trial, name, spec)  # overrides train hparams on key clash
        return hparams

    def _objective(self, trial):
        """Optuna objective: train with sampled hparams, return the metric to minimize."""
        t_start = time.time()
        print(f">>> [trial {trial.number}] {self.da_method} starting", flush=True)

        # Free GPU between trials to avoid drift across many heavy runs.
        if hasattr(self, 'algorithm'):
            del self.algorithm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fix_randomness(trial.number)

        sampled = self._sample_hparams(trial)
        prev_bs = self.hparams.get("batch_size")
        self.hparams = {**self.hparams, **sampled}

        # Only reload data if batch_size changed (avoids re-reading data each trial).
        if self.hparams.get("batch_size") != prev_bs:
            self.load_data()

        experiment_name = f"sweep_{self.source_dataset}_to_{self.target_dataset}_{self.exp_name}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{self.da_method}_trial_{trial.number}"):
            mlflow.log_params(sampled)
            mlflow.log_params({
                "source_dataset": self.source_dataset,
                "target_dataset": self.target_dataset,
                "backbone": self.backbone,
                "da_method": self.da_method,
            })

            table_results = pd.DataFrame(columns=self.results_columns)
            table_risks = pd.DataFrame(columns=self.risks_columns)

            for run_id in range(self.num_runs):
                fix_randomness(run_id)
                self.logger, self.scenario_log_dir = starting_logs(
                    self.experiment_description, self.da_method, self.exp_log_dir,
                    self.source_dataset, self.target_dataset, run_id)
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                self.initialize_algorithm()
                self.last_model, self.best_model = self.algorithm.update(
                    self.src_train_dl, self.trg_train_dl, self.loss_avg_meters,
                    self.src_val_dl, self.logger)

                self.save_checkpoint(self.home_path, self.scenario_log_dir,
                                     self.last_model, self.best_model)

                trg_acc, trg_f1, trg_auroc, src_acc, src_f1, src_auroc = self.calculate_metrics()
                metrics = (trg_acc, trg_f1, trg_auroc)
                src_risk, trg_risk = self.calculate_risks()

                scenario = f"{self.source_dataset}_to_{self.target_dataset}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, (src_risk, trg_risk))

                mlflow.log_metrics(_safe_metrics({
                    f"run_{run_id}/acc": trg_acc,
                    f"run_{run_id}/f1_score": trg_f1,
                    f"run_{run_id}/auroc": trg_auroc,
                    f"run_{run_id}/src_risk": src_risk,
                    f"run_{run_id}/trg_risk": trg_risk,
                }))

            # averages across runs
            summary = {
                "avg_acc": table_results["acc"].mean(),
                "avg_f1_score": table_results["f1_score"].mean(),
                "avg_auroc": table_results["auroc"].mean(),
                "avg_src_risk": table_risks["src_risk"].mean(),
                "avg_trg_risk": table_risks["trg_risk"].mean(),
            }
            mlflow.log_metrics(_safe_metrics(summary))

            table_results = self.add_mean_std_table(table_results, self.results_columns)
            table_risks = self.add_mean_std_table(table_risks, self.risks_columns)
            self.save_tables_to_file(table_results, f'sweep_trial_{trial.number}_results')
            self.save_tables_to_file(table_risks, f'sweep_trial_{trial.number}_risks')

        print(f">>> [trial {trial.number}] {self.da_method} done in {time.time() - t_start:.1f}s", flush=True)

        # Optuna minimizes; negate "higher-is-better" metrics.
        metric = self.metric_to_minimize
        if metric == 'src_risk':
            return summary["avg_src_risk"]
        if metric == 'f1_score':
            return -summary["avg_f1_score"]
        if metric == 'acc':
            return -summary["avg_acc"]
        if metric == 'auroc':
            return -summary["avg_auroc"]
        return summary["avg_trg_risk"]  # default: trg_risk
