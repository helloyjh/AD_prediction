import numpy as np
import itertools
from ml_models_kfoldcv import ML_Models
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils import make_result, pred_to_binary


def Trainer(config, data_x, data_y):
    seed = config.seed
    n_cores = config.n_cores

    model_name = config.model_name
    kfold = config.kfold
    cutoff = config.cutoff
    scoring = config.scoring


    x_np = np.array(data_x)
    y_np = np.array(data_y)

    kfold = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)

    true_fold = []; pred_probs = []
    accs = []; sens = []; spes = []; aucs = []

    for train_index, test_index in kfold.split(x_np[:, 2:], list(y_np[:, 4].astype('int'))):
        true_fold.append(y_np[:, 4][test_index].astype('int'))

        train_X, test_X = x_np[:, 2:][train_index], x_np[:, 2:][test_index]
        train_Y, test_Y = y_np[:, 4][train_index].astype('int'), y_np[:, 4][test_index].astype('int')

        # print(train_X)
        # print(train_Y)

        Models = ML_Models(X=train_X, Y=train_Y, scoring=scoring, n_cores=n_cores)

        if model_name == 'xgb':
            model = Models.ml_xgb()
        elif model_name == 'svm':
            model = Models.ml_svm()
        elif model_name == 'rf':
            model = Models.ml_rf()
        elif model_name == 'logistic':
            model = Models.ml_logistic()
        elif model_name == 'lgb':
            model = Models.ml_lightgbm()

        prob = model.predict_proba(test_X)[:,1]
        pred_probs.append(prob)

    true_fold = list(itertools.chain(*true_fold))
    pred_probs_fold = list(itertools.chain(*pred_probs))

    preds_fold, cutoff = pred_to_binary(true_fold, pred_probs_fold, threshold=cutoff)
    acc, sen, spe, auc = make_result(true_fold, pred_probs_fold, preds_fold)

    accs.append(acc)
    sens.append(sen)
    spes.append(spe)
    aucs.append(auc)

    return [accs, sens, spes, aucs, cutoff]

def Trainer_all(config, train_x, train_y, test_x):
    kfold = config.kfold
    seed = config.seed
    model_name = config.model_name
    n_cores = config.n_cores
    scoring = config.scoring
    cutoff = config.cutoff


    x_np = np.array(train_x)
    y_np = np.array(train_y)

    Models = ML_Models(X=x_np[:, 2:], Y=y_np[:, 4].astype('int'), cv=kfold, scoring=scoring, n_cores=n_cores)

    if model_name == 'xgb':
        model = Models.ml_xgb()
    elif model_name == 'svm':
        model = Models.ml_svm()
    elif model_name == 'rf':
        model = Models.ml_rf()
    elif model_name == 'logistic':
        model = Models.ml_logistic()
    elif model_name == 'lgb':
        model = Models.ml_lightgbm()

    probs = model.predict_proba(np.array(test_x)[:, 2:])[:, 1]
    preds, _ = pred_to_binary(y_np[:2], probs, threshold=cutoff)

    return np.array(test_x)[:,0], np.array(test_x)[:,1], probs, preds