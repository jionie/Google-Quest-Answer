from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from functools import partial
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from metric import *
from file import *
from include import *
import pickle

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
seed_everything(seed=42)

TARGET_COLUMNS = ['question_asker_intent_understanding', 'question_body_critical', 'question_conversational',
                     'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer',
                     'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent',
                     'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice',
                     'question_type_compare', 'question_type_consequence', 'question_type_definition',
                     'question_type_entity', 'question_type_instructions', 'question_type_procedure',
                     'question_type_reason_explanation', 'question_type_spelling', 'question_well_written',
                     'answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance',
                     'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure',
                     'answer_type_reason_explanation', 'answer_well_written']

# 5 classes
type_one_column_list = [
       'question_conversational', \
       'question_expect_short_answer', \
       'question_fact_seeking', \
       'question_has_commonly_accepted_answer', \
       'question_multi_intent', \
       'question_not_really_a_question', \
       'question_opinion_seeking', \
       'question_type_choice', \
       'question_type_compare', \
       'question_type_consequence', \
       'question_type_definition', \
       'question_type_entity', \
       'question_type_instructions', \
       'question_type_procedure', \
       'question_type_reason_explanation', \
       'answer_type_instructions', \
       'answer_type_procedure', \
       'answer_type_reason_explanation', \
    ]

# 3 classes
type_two_column_list = [
        'question_type_spelling'
    ]

# 9 classes
type_three_column_list = [
       'question_asker_intent_understanding', \
       'question_body_critical', \
       'question_interestingness_others', \
       'question_interestingness_self', \
       'question_well_written', \
       'answer_helpful', \
       'answer_level_of_information', \
       'answer_plausible', \
       'answer_relevance', \
       'answer_well_written'
    ]

# 17 classes
type_four_column_list = [
        'answer_satisfaction'
    ]


def apply_threshold(oof_df, column, thresholds):
    
    if column in type_one_column_list:
        
        oof_df.loc[oof_df[column] <= thresholds[0], column] = -0
        oof_df.loc[(oof_df[column] > thresholds[0]) & (oof_df[column] <= thresholds[1]), column] = -0.333333
        oof_df.loc[(oof_df[column] > thresholds[1]) & (oof_df[column] <= thresholds[2]), column] = -0.500000
        oof_df.loc[(oof_df[column] > thresholds[2]) & (oof_df[column] <= thresholds[3]), column] = -0.666667
        oof_df.loc[(oof_df[column] > thresholds[3]), column] = -1
        
        oof_df.loc[:, column] = -1 * oof_df.loc[:, column]
        
    elif column in type_two_column_list:
        
        oof_df.loc[oof_df[column] <= thresholds[0], column] = -0
        oof_df.loc[(oof_df[column] > thresholds[0]) & (oof_df[column] <= thresholds[1]), column] = -0.333333
        oof_df.loc[(oof_df[column] > thresholds[1]), column] = -0.666667
        
        oof_df.loc[:, column] = -1 * oof_df.loc[:, column]
        
    elif column in type_three_column_list:
        
        oof_df.loc[oof_df[column] <= thresholds[0], column] = -0.333333
        oof_df.loc[(oof_df[column] > thresholds[0]) & (oof_df[column] <= thresholds[1]), column] = -0.444444
        oof_df.loc[(oof_df[column] > thresholds[1]) & (oof_df[column] <= thresholds[2]), column] = -0.5
        oof_df.loc[(oof_df[column] > thresholds[2]) & (oof_df[column] <= thresholds[3]), column] = -0.555556
        oof_df.loc[(oof_df[column] > thresholds[3]) & (oof_df[column] <= thresholds[4]), column] = -0.666667
        oof_df.loc[(oof_df[column] > thresholds[4]) & (oof_df[column] <= thresholds[5]), column] = -0.777778
        oof_df.loc[(oof_df[column] > thresholds[5]) & (oof_df[column] <= thresholds[6]), column] = -0.833333
        oof_df.loc[(oof_df[column] > thresholds[6]) & (oof_df[column] <= thresholds[7]), column] = -0.888889
        oof_df.loc[(oof_df[column] > thresholds[7]), column] = -1
        
        oof_df.loc[:, column] = -1 * oof_df.loc[:, column]
        
    elif column in type_four_column_list:
        
        oof_df.loc[oof_df[column] <= thresholds[0], column] = -0.200000
        oof_df.loc[(oof_df[column] > thresholds[0]) & (oof_df[column] <= thresholds[1]), column] = -0.266667
        oof_df.loc[(oof_df[column] > thresholds[1]) & (oof_df[column] <= thresholds[2]), column] = -0.300000
        oof_df.loc[(oof_df[column] > thresholds[2]) & (oof_df[column] <= thresholds[3]), column] = -0.333333
        oof_df.loc[(oof_df[column] > thresholds[3]) & (oof_df[column] <= thresholds[4]), column] = -0.400000
        oof_df.loc[(oof_df[column] > thresholds[4]) & (oof_df[column] <= thresholds[5]), column] = -0.466667
        oof_df.loc[(oof_df[column] > thresholds[5]) & (oof_df[column] <= thresholds[6]), column] = -0.500000
        oof_df.loc[(oof_df[column] > thresholds[6]) & (oof_df[column] <= thresholds[7]), column] = -0.533333
        oof_df.loc[(oof_df[column] > thresholds[7]) & (oof_df[column] <= thresholds[8]), column] = -0.600000
        oof_df.loc[(oof_df[column] > thresholds[8]) & (oof_df[column] <= thresholds[9]), column] = -0.666667
        oof_df.loc[(oof_df[column] > thresholds[9]) & (oof_df[column] <= thresholds[10]), column] = -0.700000
        oof_df.loc[(oof_df[column] > thresholds[10]) & (oof_df[column] <= thresholds[11]), column] = -0.733333
        oof_df.loc[(oof_df[column] > thresholds[11]) & (oof_df[column] <= thresholds[12]), column] = -0.800000
        oof_df.loc[(oof_df[column] > thresholds[12]) & (oof_df[column] <= thresholds[13]), column] = -0.866667
        oof_df.loc[(oof_df[column] > thresholds[13]) & (oof_df[column] <= thresholds[14]), column] = -0.900000
        oof_df.loc[(oof_df[column] > thresholds[14]) & (oof_df[column] <= thresholds[15]), column] = -0.933333
        oof_df.loc[(oof_df[column] > thresholds[15]), column] = -1
        
        oof_df.loc[:, column] = -1 * oof_df.loc[:, column]
        
    else:
        raise NotImplementedError 
    
    return oof_df

def get_best_threshold(column, oof_df, train_df, test_size=10000):
    
    
    print(column)
    if column in type_one_column_list:
        num_bin = 5 - 1
        original_thresholds = [0.16667, 0.41667, 0.58333, 0.73333]
    elif column in type_two_column_list:
        num_bin = 3 - 1
        original_thresholds = [0.16667, 0.5]
    elif column in type_three_column_list:
        num_bin = 9 - 1
        original_thresholds = [0.388889, 0.472222, 0.527778, 0.611111, 0.722222, 0.805556, 0.861111, 0.944445]
    elif column in type_four_column_list:
        num_bin = 17 - 1
        original_thresholds = [0.233333, 0.283333, 0.316666, 0.366666, 0.433333, 0.483333, 0.516666, 0.566666, \
                               0.633333, 0.683333, 0.716666, 0.767666, 0.833333, 0.883333, 0.916666, 0.966666
                              ]
    else:
        raise NotImplementedError
    
    oof_df_copy = oof_df.copy()
    spearman_no_threshold = Spearman(train_df[TARGET_COLUMNS].values, oof_df_copy[TARGET_COLUMNS].values)
    
    # apply minmax scaler
    scaler = MinMaxScaler()
    oof_df_column_values = oof_df_copy[column].values.reshape(-1, 1)
    oof_df_copy[column] = scaler.fit_transform(oof_df_column_values).squeeze()
    
    oof_df_copy = apply_threshold(oof_df_copy, column, original_thresholds)
    spearman_original = Spearman(train_df[TARGET_COLUMNS].values, oof_df_copy[TARGET_COLUMNS].values)
    
    if spearman_original > spearman_no_threshold:
        best_spearman = spearman_original
        best_thresholds = original_thresholds
        best_oof_df = oof_df_copy
    else:
        best_spearman = spearman_no_threshold
        best_thresholds = []
        best_oof_df = oof_df
    
    log = Logger()
    log.open(column + '.txt', mode='a+')
    
    log.write('oof_spearman :%f\n' % \
        (best_spearman))
    log.write('\n')
    
    for i in range(test_size):
    
        if (i % 10 == 0):
            print("processing", i, "of", test_size)
        
        thresholds = list(np.random.uniform(0, 1, num_bin))
        thresholds.sort()
        
        oof_df_copy = oof_df.copy()
        oof_df_copy = apply_threshold(oof_df_copy, column, thresholds)
        spearman_proposed = Spearman(train_df[TARGET_COLUMNS].values, oof_df_copy[TARGET_COLUMNS].values)
    
        if (spearman_proposed > best_spearman):
            log.write('oof_spearman increase from %f to % f\n' % \
                (best_spearman, spearman_proposed))
            log.write('\n')
            best_thresholds = thresholds
            best_spearman = spearman_proposed
            best_oof_df = oof_df_copy
            
    return spearman_original, best_spearman, best_thresholds, best_oof_df


if __name__ == "__main__":
    
    model_folder = "/media/jionie/my_disk/Kaggle/Google_Quest_Answer/model/"
    oof_question_answer = pd.read_csv(model_folder + "bert/bert-base-uncased-Answer-bce-BertAdam-WarmupLinearSchedule-5-2020-aug_differential/oof_question_answer.csv")
    oof_xlnet_base_cased = pd.read_csv(model_folder + "xlnet/xlnet-base-cased-bce-BertAdam-WarmupLinearSchedule-5-1997-aug_differential_relu_v1/oof_xlnet_base_cased.csv")
    oof_bert_base_uncased = pd.read_csv(model_folder + "bert/bert-base-uncased-bce-BertAdam-WarmupLinearSchedule-10-2020-aug_differential_relu_v2/oof_bert_base_uncased.csv")
    oof_bert_base_cased = pd.read_csv(model_folder + "bert/bert-base-cased-bce-BertAdam-WarmupLinearSchedule-10-1996-aug_differential_relu_v2/oof-bert-base-cased-v2.csv")

    oof_df = (oof_xlnet_base_cased[TARGET_COLUMNS] + oof_bert_base_uncased[TARGET_COLUMNS] + oof_bert_base_cased[TARGET_COLUMNS])/3.0

    train_df = pd.read_csv("/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/train.csv")
    
    best_threshold_list = []
    best_spearman_list = []
    
    for column in TARGET_COLUMNS:
        
        _, best_spearman, best_thresholds, best_oof_df = get_best_threshold(column, oof_df, train_df, test_size=6000)
        best_threshold_list.append(best_thresholds)
        best_spearman_list.append(best_spearman)
        oof_df = best_oof_df
        
        
        with open('best_thresholds.txt', 'a+') as filehandle:
            filehandle.write('%s\n' % best_thresholds)
                
        with open('best_spearman.txt', 'a+') as filehandle:
            filehandle.write('%s\n' % best_spearman)
            
    with open("best_threshold_pickle.txt", "wb") as fp:   #Pickling
        pickle.dump(best_threshold_list, fp)
    with open("best_spearman_pickle.txt", "wb") as fp:   #Pickling
        pickle.dump(best_spearman_list, fp)