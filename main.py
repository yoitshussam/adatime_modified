from trainers.train import Trainer
import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")
from load_data import load_labelled_and_unlabelled
import argparse
parser = argparse.ArgumentParser()

if __name__ == "__main__":

    # ========  Experiments Phase ================
    parser.add_argument('--phase',               default='train',         type=str, help='train, test')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name',               default='EXP1',         type=str, help='experiment name')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method',              default='ALL',               type=str, help='NO_ADAPT, Deep_Coral, MMDA, DANN, CDAN, DIRT, DSAN, HoMM, CoDATS, AdvSKM, SASA, CoTMix, TARGET_ONLY')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path',              default=r'../dataset',                  type=str, help='Path containing datasets')
    parser.add_argument('--source_dataset',                default='RealWorld',                      type=str, help='Dataset of choice: (RealWorld - PAMAP2 - Mhealth)')
    parser.add_argument('--target_dataset',                default='Pamap2',                      type=str, help='Dataset of choice: (RealWorld - PAMAP2 - Mhealth)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone',               default='HAR_CNN',                      type=str, help='Backbone of choice: (CNN - HAR_CNN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs',               default=1,                          type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device',                 default= "cuda",                   type=str, help='cpu or cuda')
    
    methods=['Deep_Coral','DDC', 'MMDA', 'DANN', 'CDAN', 'DIRT', 'DSAN', 'HoMM', 'CoDATS', 'AdvSKM','SASA', 'CoTMix','SWL_Adapt',"uDAR","DAAN","ACON","RAINCOAT","SSSS_TSA","CLUDA"]
    # arguments
    #'NO_ADAPT',
    # methods=["DAAN","ACON","RAINCOAT","SSSS_TSA","CLUDA"]

    args = parser.parse_args()

    if(args.da_method=='ALL'):

        for method in methods:

            args.da_method=method
            trainer=Trainer(args)
            trainer.fit()
    else:

        trainer=Trainer(args)
        trainer.fit()
