from trainers.sweep import Trainer
import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
import argparse
parser = argparse.ArgumentParser()


if __name__ == "__main__":
    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='Deep_Coral', type=str,
                        help='DANN, Deep_Coral, MMDA, DIRT, CDAN, DSAN, HoMM, CoDATS, '
                             'AdvSKM, DDC, SASA, CoTMix, SWL_Adapt, uDAR, DAAN, ACON, '
                             'RAINCOAT, SSSS_TSA, CLUDA, ALL')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'../dataset', type=str, help='Path containing datasets')
    parser.add_argument('--source_dataset', default='RealWorld_Male', type=str, help='Source dataset (sweep split)')
    parser.add_argument('--target_dataset', default='RealWorld_Female', type=str, help='Target dataset (sweep split)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - HARCNN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=1, type=int, help='Number of consecutive runs with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument('--exp_name', default='sweep_EXP1', type=str, help='experiment name')

    # ======== Sweep settings (Optuna) ============
    parser.add_argument('--num_sweeps', default=10, type=int, help='Number of Optuna trials')
    parser.add_argument('--hp_search_strategy', default="bayes", type=str,
                        help='Optuna sampler: random, grid, bayes (TPE)')
    parser.add_argument('--metric_to_minimize', default="trg_risk", type=str,
                        help='Metric to optimize: trg_risk, src_risk (minimized), '
                             'or f1_score, acc, auroc (maximized).')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='experiments_logs/sweep_logs', type=str,
                        help='Directory containing all sweep experiments')

    methods = ['Deep_Coral', 'DDC', 'MMDA', 'DANN', 'CDAN', 'DIRT', 'DSAN', 'HoMM',
               'CoDATS', 'AdvSKM', 'SASA', 'CoTMix', 'SWL_Adapt', 'uDAR', 'DAAN',
               'ACON', 'RAINCOAT', 'SSSS_TSA', 'CLUDA']

    args = parser.parse_args()
    if args.da_method == 'ALL':
        for method in methods:
            args.da_method = method
            trainer = Trainer(args)
            trainer.sweep()
    else:
        trainer = Trainer(args)
        trainer.sweep()
