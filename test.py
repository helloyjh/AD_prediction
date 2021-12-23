from data_loader import Data_loader
from trainer import Trainer_all

import pandas as pd
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--kfold', type=int, default=10)
parser.add_argument('--n_cores', type=int, default=10)
parser.add_argument('--cutoff', type=int, default=0.5)
parser.add_argument('--scoring', type=str, default=None)

parser.add_argument('--model_name', type=str, default='logistic')
parser.add_argument('--score_type', type=str, default='raw')

parser.add_argument('--csv_path', type=str, default='/tf/storage/nrcdbox/data/csv/wgee_project/all_mri_snsb_preprocessing_211029.csv')
parser.add_argument('--save_path', type=str, default='/tf/storage/result/ml/')

config = parser.parse_args()
print(config)

if __name__ == '__main__':
    print('##data loading##')
    train_x, train_y, test_x, test_y = Data_loader(config)
    print('##training##')
    oid, mriid, probs, preds = Trainer_all(config, train_x, train_y, test_x)

    results = pd.DataFrame({'OID': oid, 'MRI_ID': mriid,'probs': probs,'preds' : preds})
    results.to_csv(config.save_path + 'ML_results_MCI_probs_{}_{}.csv'.format(config.model_name, config.seed), index=False, mode='w')