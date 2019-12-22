import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata


def Spearman(y_val, y_pred):

    spearman = []
    # y_val = y_val + np.random.normal(0, 1e-8, y_val.shape)
    # y_pred = y_pred + np.random.normal(0, 1e-8, y_pred.shape)

    for ind in range(y_pred.shape[1]):
        tmp_spearman, _ = spearmanr(y_val[:, ind], y_pred[:, ind])
        spearman.append(tmp_spearman)
    
    return np.mean(spearman)


def Spearman_v2(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues + np.random.normal(0, 1e-7, col_pred.shape[0]), col_pred).correlation)
    return np.mean(rhos)