import pandas as pd
import numpy as np
import sys
import sys
from utils import rank_INT


def Data_loader(config):
    path=config.csv_path

    info = pd.read_csv(path)

    info = info[info['MRI_ID'] > 14000000]
    #info = info[(info['diagnosis']=='CN')|(info['diagnosis']=='AD')]

    info['PHENO_3group'] = -9
    info.loc[info['diagnosis'] == 'CN', 'PHENO_3group'] = 0
    info.loc[info['diagnosis'] == 'AD', 'PHENO_3group'] = 1
    info.loc[(info['diagnosis'] != 'CN')&(info['diagnosis'] != 'AD'), 'PHENO_3group'] = 2

    info = info[info['Apoe'] != '0/0']
    info = info[info['Apoe'] != 'missing']
    info = info[info['Apoe'] != 'Missing']

    info['Apoe4'] = -9
    info.loc[info['Apoe'] == 'E2/E2', 'Apoe4'] = 0
    info.loc[info['Apoe'] == 'E2/E3', 'Apoe4'] = 0
    info.loc[info['Apoe'] == 'E3/E3', 'Apoe4'] = 0
    info.loc[info['Apoe'] == 'E2/E4', 'Apoe4'] = 1
    info.loc[info['Apoe'] == 'E3/E4', 'Apoe4'] = 1
    info.loc[info['Apoe'] == 'E4/E4', 'Apoe4'] = 2

    info = info.reset_index()

    mri_thick = [col for col in list(info.columns) if 'L_' in col] + [col for col in list(info.columns) if
                                                                      'R_' in col] + ['LThickness', 'RThickness']
    mri_vol = ['LLatVent', 'RLatVent', 'Lthal', 'Rthal', 'Lcaud', 'Rcaud', 'Lput', 'Rput', 'Lpal', 'Rpal', 'Lhippo',
               'Rhippo', 'Lamyg', 'Ramyg', 'Laccumb', 'Raccumb', 'ICV']
    mri = mri_thick + mri_vol
    demo = ['Apoe4', 'Gender', 'YearsOfEducation']
    ids = ['OID', 'MRI_ID']
    pheno = ['diagnosis_orig', 'diagnosis', 'PHENO_3group']
    score = ['SNSB_II_Domain_Attention', 'SNSB_II_Domain_Language', 'SNSB_II_Domain_Visuospatial',
             'SNSB_II_Domain_Memory', 'SNSB_II_Domain_Frontal', 'K_MMSE_total_score']

    if config.score_type == 'z':
        score = [ i + '_z' for i in score]
        info = info[ids + demo + score + mri + pheno]
        info = info.dropna(axis=0)

    elif config.score_type == 'raw':
        info = info[ids + demo + score + mri + pheno]
        info = info.dropna(axis=0)
        #score INT
        info.loc[:, score] = info.loc[:, score].apply(rank_INT, axis=0)

    # mri INT
    info.loc[:, mri] = info.loc[:, mri].apply(rank_INT, axis=0)

    #age, eductaion minmax
    #info.loc[:, 'Age'] = (info.Age - min(info.Age)) / (max(info.Age) - min(info.Age))
    info.loc[:, 'YearsOfEducation'] = (info.YearsOfEducation - min(info.YearsOfEducation)) / (
                max(info.YearsOfEducation) - min(info.YearsOfEducation))

    cn_oid = []
    for i in set(info.OID):
        a = list(set(list(info.loc[info.OID == i, 'diagnosis'])))
        if ((len(a) == 1) & (a[0] == 'CN')):
            cn_oid.append(i)

    train_x_y = info[(info['diagnosis']=='AD')|(info['OID'].isin(cn_oid))]
    #train_x_y = info[info['PHENO_3group']!=2]
    test_x_y = info[info['PHENO_3group']==2]

    train_x = train_x_y[ids + demo + score + mri]
    train_y = train_x_y[ids + pheno]

    test_x = test_x_y[ids + demo + score + mri]
    test_y = test_x_y[ids + pheno]

    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)

    test_x = test_x.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)

    print(train_x.shape, train_y.shape)

    return train_x, train_y, test_x, test_y


# def Data_loader(config):
#     path=config.csv_path
#
#     info = pd.read_csv(path)
#
#     info = info[info['PHENO_3group'] > -1]
#     info = info[info['Apoe'] != '0/0']
#     info = info[info['Apoe'] != 'missing']
#     info = info[info['Apoe'] != 'Missing']
#
#     info['Apoe4'] = -9
#     info.loc[info['Apoe'] == 'E2/E2', 'Apoe4'] = 0
#     info.loc[info['Apoe'] == 'E2/E3', 'Apoe4'] = 0
#     info.loc[info['Apoe'] == 'E3/E3', 'Apoe4'] = 0
#     info.loc[info['Apoe'] == 'E2/E4', 'Apoe4'] = 1
#     info.loc[info['Apoe'] == 'E3/E4', 'Apoe4'] = 1
#     info.loc[info['Apoe'] == 'E4/E4', 'Apoe4'] = 2
#
#     info = info.reset_index()
#
#     mri_thick = [col for col in list(info.columns) if 'L_' in col] + [col for col in list(info.columns) if 'R_' in col] + ['LThickness', 'RThickness']
#     mri_vol = ['LLatVent', 'RLatVent', 'Lthal', 'Rthal', 'Lcaud', 'Rcaud', 'Lput', 'Rput', 'Lpal', 'Rpal', 'Lhippo', 'Rhippo', 'Lamyg', 'Ramyg', 'Laccumb', 'Raccumb', 'ICV']
#     mri = mri_thick + mri_vol
#     demo = ['Apoe4', 'Sex', 'YearsOfEducation']
#     ids = ['OID', 'MRI_ID']
#     pheno = ['PHENO_3group']
#     score = ['SNSB_II_Domain_Attention', 'SNSB_II_Domain_Language', 'SNSB_II_Domain_Visuospatial',
#              'SNSB_II_Domain_Memory', 'SNSB_II_Domain_Frontal', 'K_MMSE_total_score']
#
#     if config.score_type == 'z':
#         score = [ i + '_z' for i in score]
#         info = info[ids + demo + score + mri + pheno]
#         info = info.dropna(axis=0)
#
#     elif config.score_type == 'raw':
#         info = info[ids + demo + score + mri + pheno]
#         info = info.dropna(axis=0)
#         #score INT
#         info.loc[:, score] = info.loc[:, score].apply(rank_INT, axis=0)
#
#     # mri INT
#     info.loc[:, mri] = info.loc[:, mri].apply(rank_INT, axis=0)
#
#     #age, eductaion minmax
#     #info.loc[:, 'Age'] = (info.Age - min(info.Age)) / (max(info.Age) - min(info.Age))
#     info.loc[:, 'YearsOfEducation'] = (info.YearsOfEducation - min(info.YearsOfEducation)) / (
#                 max(info.YearsOfEducation) - min(info.YearsOfEducation))
#
#     train_x_y = info[info['PHENO_3group']!=2]
#     test_x_y = info[info['PHENO_3group']==2]
#
#     train_x = train_x_y[ids + demo + score + mri]
#     train_y = train_x_y[ids + pheno]
#
#     test_x = test_x_y[ids + demo + score + mri]
#     test_y = test_x_y[ids + pheno]
#
#     return train_x, train_y, test_x, test_y
