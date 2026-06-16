import sys

sys.path.append('../')
import torch
import torch.nn.functional as F
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections
import argparse
import warnings
import sklearn.exceptions

from configs.sweep_params import sweep_alg_hparams, sweep_train_hparams 

from utils import fix_randomness, starting_logs, DictAsObject
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from utils import AverageMeter

from trainers.abstract_trainer import AbstractTrainer
from dataloader.dataloader import data_generator_pt, few_shot_data_generator

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class Trainer(AbstractTrainer):
    """
    This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # sweep parameters
        self.num_sweeps = args.num_sweeps
        self.sweep_project_wandb = args.sweep_project_wandb
        self.wandb_entity = args.wandb_entity
        self.hp_search_strategy = args.hp_search_strategy # e.g., random, grid, bayes (we choose bayes)
        self.metric_to_minimize = args.metric_to_minimize # e.g., src_risk, trg_risk (we choose trg_risk)
        self.sweep_iter = 0   # counter for sweep iterations (different combinations of hyper-parameters)
        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir)
        os.makedirs(self.exp_log_dir, exist_ok=True)

    def load_data(self):
        """Override to use .pt-based loader for sweep datasets."""
        self.src_train_dl = data_generator_pt(self.source_data_path, self.dataset_configs, self.hparams, 'source', "train")
        self.src_test_dl  = data_generator_pt(self.source_data_path, self.dataset_configs, self.hparams, 'source', "test")
        self.src_val_dl   = data_generator_pt(self.source_data_path, self.dataset_configs, self.hparams, 'source', "val")

        self.trg_train_dl = data_generator_pt(self.target_data_path, self.dataset_configs, self.hparams, 'target', "train")
        self.trg_test_dl  = data_generator_pt(self.target_data_path, self.dataset_configs, self.hparams, 'target', "test")

    def sweep(self):
        # Reset counter at the start of each sweep
        self.sweep_iter = 0
        
        # sweep configurations
        sweep_config = {
            'method': self.hp_search_strategy,
            'metric': {'name': self.metric_to_minimize, 'goal': 'minimize'},
            'name': self.da_method + '_' + self.backbone,
            
            # ---(MERGE): Use BOTH param dictionaries ---
            'parameters': {
                **sweep_train_hparams,
                **sweep_alg_hparams[self.da_method]
            }
        }
        sweep_id = wandb.sweep(sweep_config, project=self.sweep_project_wandb, entity=self.wandb_entity)

        wandb.agent(sweep_id, self.train, count=self.num_sweeps)

    def train(self):
        # Increment counter
        self.sweep_iter += 1

        run = wandb.init(config=self.hparams, project=self.sweep_project_wandb)

        # Sweep name
        run_name = f"{self.da_method}_sweep_{self.sweep_iter}"
        wandb.run.name = run_name  # Set the run name explicitly in W&B
        self.hparams= wandb.config
        
        # create tables for results and risks
        columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = wandb.Table(columns=columns, allow_mixed_types=True)
        columns = ["scenario", "run", "src_risk", "trg_risk"]
        table_risks = wandb.Table(columns=columns, allow_mixed_types=True)

        for run_id in range(self.num_runs):
            # set random seed and create logger
            fix_randomness(run_id)
            self.logger, self.scenario_log_dir = starting_logs(self.experiment_description, self.da_method, self.exp_log_dir,
                                                                self.source_dataset, self.target_dataset, run_id)

            # average meters
            self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

            # load data 
            self.load_data()

            # initiate the domain adaptation algorithm
            self.initialize_algorithm()

            # Train the domain adaptation algorithm
            self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.trg_train_dl, self.loss_avg_meters,self.src_val_dl, self.logger)

            # calculate metrics and risks
            trg_acc, trg_f1, trg_auroc, src_acc, src_f1, src_auroc= self.calculate_metrics()
            metrics=(trg_acc,trg_f1,trg_auroc)

            risks = self.calculate_risks()

            # append results to tables
            scenario = f"{self.source_dataset}_to_{self.target_dataset}"
            table_results.add_data(scenario, run_id, *metrics)
            table_risks.add_data(scenario, run_id, *risks)

        # calculate overall metrics and risks
        total_results, summary_metrics = self.calculate_avg_std_wandb_table(table_results)
        total_risks, summary_risks = self.calculate_avg_std_wandb_table(table_risks)

        # log results to WandB
        self.wandb_logging(total_results, total_risks, summary_metrics, summary_risks)

        # finish the run
        run.finish()