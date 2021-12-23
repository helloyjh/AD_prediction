from data_loader import Data_loader
from trainer import Trainer

import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--kfold', type=int, default=10)
parser.add_argument('--n_cores', type=int, default=10)
parser.add_argument('--cutoff', type=int, default=0.5)
parser.add_argument('--scoring', type=str, default=None)

parser.add_argument('--model_name', type=str, default='logistic')
parser.add_argument('--score_type', type=str, default='z')


parser.add_argument('--csv_path', type=str, default='/tf/storage/nrcdbox/data/csv/wgee_project/all_mri_snsb_preprocessing_211222.csv')
parser.add_argument('--save_path', type=str, default='/tf/storage/result/ml/')

config = parser.parse_args()
print(config)

if __name__ == '__main__':
    print('##data loading##')
    train_x, train_y, _, _ = Data_loader(config)
    print('##training##')
    model_results = Trainer(config, train_x, train_y)

    results = [config.seed] + [config.model_name] + model_results
    results_pd = pd.DataFrame(data=[results], columns=['seed', 'model_name', 'accs', 'sens', 'spes', 'aucs', 'cutoff'])

    if not os.path.exists(config.save_path + 'ML_results_211222.csv'):
        results_pd.to_csv(config.save_path + 'ML_results_211222.csv', index=False, mode='w')
    else:
        results_pd.to_csv(config.save_path + 'ML_results_211222.csv', index=False, mode='a', header=False)