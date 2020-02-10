import pandas as pd
import numpy as np
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Lasso, Ridge, ElasticNet

def Spearman(y_val, y_pred):

    spearman = []
    # y_val = y_val + np.random.normal(0, 1e-8, y_val.shape)
    # y_pred = y_pred + np.random.normal(0, 1e-8, y_pred.shape)

    for ind in range(y_pred.shape[1]):
        tmp_spearman, _ = spearmanr(y_val[:, ind], y_pred[:, ind])
        spearman.append(tmp_spearman)
    spearman = np.nan_to_num(spearman)
    return np.mean(spearman), spearman


data = pd.read_csv("/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/train.csv")

TARGET_COLUMNS = data.columns[11:]

data.columns

model_folder = "/media/jionie/my_disk/Kaggle/Google_Quest_Answer/model/"
model_1 = pd.read_csv(model_folder + "oof_bert_base_cased_swa_5_fold.csv")
model_2 = pd.read_csv(model_folder + "oof_bert_base_cased_two_model_swa.csv")
model_3 = pd.read_csv(model_folder + "oof_bert_base_uncased_swa_5_fold.csv")
model_4 = pd.read_csv(model_folder + "oof_bert_base_uncased_two_model_swa.csv")
model_5 = pd.read_csv(model_folder + "oof_xlnet_base_cased.csv")
model_6 = pd.read_csv(model_folder + "oof_xlnet_base_cased_two_model_swa.csv")
model_7 = pd.read_csv(model_folder + "oof_roberta_base_two_model_swa.csv")
model_8 = pd.read_csv(model_folder + "oof_roberta_base_swa.csv")


ensemble_cv = (model_1[TARGET_COLUMNS] + model_2[TARGET_COLUMNS] + model_3[TARGET_COLUMNS] + model_4[TARGET_COLUMNS] + model_5[TARGET_COLUMNS] + model_6[TARGET_COLUMNS] + model_7[TARGET_COLUMNS] + model_8[TARGET_COLUMNS]) / 8

print(Spearman(ensemble_cv.values, data[TARGET_COLUMNS].values))

frames = []
frames.append(model_1[TARGET_COLUMNS].values)
frames.append(model_2[TARGET_COLUMNS].values)
frames.append(model_3[TARGET_COLUMNS].values)
frames.append(model_4[TARGET_COLUMNS].values)
frames.append(model_5[TARGET_COLUMNS].values)
frames.append(model_6[TARGET_COLUMNS].values)
frames.append(model_7[TARGET_COLUMNS].values)
frames.append(model_8[TARGET_COLUMNS].values)

X_meta = np.concatenate(frames,axis=1)

X_meta.shape

TARGET_COLUMNS = data.columns[11:]

y = data[TARGET_COLUMNS].values

train = shuffle(data, random_state=333)

folds = GroupKFold(n_splits=5).split(X=train.question_title, groups=train.question_title)

stck_oof = np.zeros((len(train),len(TARGET_COLUMNS)))

for fold, (train_idx, valid_idx) in enumerate(folds):
    print(fold)
    stck_train = X_meta[train_idx]
    stck_valid = X_meta[valid_idx]
    
    label_train = y[train_idx]
    label_valid = y[valid_idx]
    
    #model = MultiTaskElasticNet(alpha=0.0001, random_state=42)
    model = Ridge(alpha=40, random_state=50, max_iter=2000)
    #model_2 = Lasso(alpha=0.0001, random_state=42, max_iter=2000)
    #model_3 = ElasticNet(alpha=0.0001, random_state=42, max_iter=2000)
    model.fit(stck_train, label_train)
    #model_2.fit(stck_train, label_train)
    #model_3.fit(stck_train, label_train)
    stck_oof[valid_idx] = model.predict(stck_valid)#(model_3.predict(stck_valid) + model.predict(stck_valid) + model_2.predict(stck_valid))/3.0
    

stck_oof[ stck_oof > 1.0] = 1.0
stck_oof[ stck_oof < 0.0] = 0.0

print(Spearman(stck_oof, data[TARGET_COLUMNS].values))