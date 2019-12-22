import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata


def Spearman(y_val, y_pred):

    spearman = []

    for ind in range(y_pred.shape[1]):
        tmp_spearman, _ = spearmanr(y_val[:, ind] + np.random.normal(0, 1e-7, y_val.shape[0]), y_pred[:, ind] + np.random.normal(0, 1e-7, y_pred.shape[0]))
        spearman.append(tmp_spearman)
    
    return np.mean(spearman)