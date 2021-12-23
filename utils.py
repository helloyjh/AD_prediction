import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

def rank_INT(series, c=3.0 / 8, stochastic=True):
    """ Perform rank-based inverse normal transformation on pandas series.
        If stochastic is True ties are given rank randomly, otherwise ties will
        share the same value. NaN values are ignored.
        Args:
            param1 (pandas.Series):   Series of values to transform
            param2 (Optional[float]): Constand parameter (Bloms constant)
            param3 (Optional[bool]):  Whether to randomise rank of ties

        Returns:
            pandas.Series
    """

    # Check input
    assert (isinstance(series, pd.Series))
    assert (isinstance(c, float))
    assert (isinstance(stochastic, bool))

    # Set seed
    np.random.seed(123)

    # Take original series indexes
    orig_idx = series.index

    # Drop NaNs
    series = series.loc[~pd.isnull(series)]

    # Get ranks
    if stochastic == True:
        # Shuffle by index
        series = series.loc[np.random.permutation(series.index)]
        # Get rank, ties are determined by their position in the series (hence
        # why we randomised the series)
        rank = ss.rankdata(series, method="ordinal")
    else:
        # Get rank, ties are averaged
        rank = ss.rankdata(series, method="average")

    # Convert numpy array back to series
    rank = pd.Series(rank, index=series.index)

    # Convert rank to normal distribution
    transformed = rank.apply(rank_to_normal, c=c, n=len(rank))

    return transformed[orig_idx]


def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2 * c + 1)
    return ss.norm.ppf(x)


def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def pred_to_binary(target_array, pred_array, threshold=0.5):
    if threshold == 0.5:
        # pred_binary = sorted(list(pred_array))
        # threshold = pred_binary[int(len(pred_binary) * per_of_zero / 100)]

        pred_binary = np.copy(pred_array)
        pred_binary[pred_binary > threshold] = 1
        pred_binary[pred_binary <= threshold] = 0

    else:
        threshold = Find_Optimal_Cutoff(target_array, pred_array)
        pred_binary = np.copy(pred_array)
        pred_binary[pred_binary > threshold] = 1
        pred_binary[pred_binary <= threshold] = 0

    return pred_binary, threshold

# def model_test(model, S_test, test_Y):
#     prob = model.predict_proba(S_test)[:, 1]
#     pred = pred_to_binary(test_Y, prob)
#
#     tn, fp, fn, tp = confusion_matrix(test_Y, pred).ravel()
#     sen = tp / (tp + fn)
#     spe = tn / (tn + fp)
#     acc = (tp + tn) / (tn + fp + fn + tp)
#     return [acc, sen, spe, pred]

def make_result(trues, probs, preds):

    tn, fp, fn, tp = confusion_matrix(trues, preds).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    acc = (tp + tn) / (tn + fp + fn + tp)
    auc = roc_auc_score(trues, probs)

    return [acc, sen, spe, auc]

def make_ensemble_result(prob, test_Y):
    pred = pred_to_binary(test_Y, prob)

    tn, fp, fn, tp = confusion_matrix(test_Y, pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    acc = (tp + tn) / (tn + fp + fn + tp)
    return [acc, sen, spe, prob, pred, list(test_Y)]

def minmax(x):
    high = np.max(x, axis=0)
    low = np.min(x, axis=0)

    x_minmax = (x - low) / (high - low)

    return x_minmax


def neuroi_preprocessing(r,type):
    roi = pd.read_csv('/tf/storage/nrcdbox/data/csv/roi_info.csv')
    roi = roi[roi.use == 'o']

    p = r[r['mtype'] == 'P']
    t = r[r['mtype'] == 'T']
    v = r[r['mtype'] == 'V']
    mriid = p['MRI_ID']

    col_v = list(roi[roi['col'] == 'V']['column_no'])
    col_t = list(roi[roi['col'] == 'T']['column_no'])

    x_v = ['X' + x.split('c')[1] for x in col_v]
    x_t = ['X' + x.split('c')[1] for x in col_t]

    p = p[x_v]
    t = t[x_t]
    v = v[x_v]

    mriid = mriid.reset_index(drop=True)
    p = p.reset_index(drop=True)
    t = t.reset_index(drop=True)
    v = v.reset_index(drop=True)

    p.columns = ['P_' + i for i in list(p.columns)]
    t.columns = ['T_' + i for i in list(t.columns)]
    v.columns = ['V_' + i for i in list(v.columns)]

    if type == 'TV':
        final = pd.concat([mriid, t, v], axis=1)
    elif type == 'PTV':
        final = pd.concat([mriid, t, v, p], axis=1)

    return final