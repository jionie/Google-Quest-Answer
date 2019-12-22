import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata


def Spearman(y_val, y_pred):

    spearman = [ spearmanr(np.squeeze(y_val[:, ind]) + 1e-8, np.squeeze(y_pred[:, ind]) + 1e-8).correlation for ind in range(y_pred.shape[1]) ]

    return np.mean(spearman)