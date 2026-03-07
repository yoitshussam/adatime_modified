#!/bin/bash
python main.py --num_runs 10 --source_dataset Pamap2 --target_dataset RealWorld --data_path ../dataset --da_method ALL --exp_name EXP_2
python main.py --num_runs 10 --source_dataset mhealth --target_dataset RealWorld --data_path ../dataset --da_method ALL --exp_name EXP_2
python main.py --num_runs 10 --source_dataset mhealth --target_dataset Pamap2 --data_path ../dataset --da_method ALL --exp_name EXP_2
python main.py --num_runs 10 --source_dataset RealWorld --target_dataset mhealth --data_path ../dataset --da_method ALL --exp_name EXP_2
python main.py --num_runs 10 --source_dataset Pamap2 --target_dataset mhealth --data_path ../dataset --da_method ALL --exp_name EXP_2
