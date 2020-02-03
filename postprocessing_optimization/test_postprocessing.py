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


def postprocessing(oof_df, target_columns):
    
    scaler = MinMaxScaler()
    
    # type 1 column [0, 0.333333, 0.5, 0.666667, 1]
    # type 2 column [0, 0.333333, 0.666667]
    # type 3 column [0.333333, 0.444444, 0.5, 0.555556, 0.666667, 0.777778, 0.8333333, 0.888889, 1]
    # type 4 column [0.200000, 0.266667, 0.300000, 0.333333, 0.400000, \
    # 0.466667, 0.5, 0.533333, 0.600000, 0.666667, 0.700000, \
    # 0.733333, 0.800000, 0.866667, 0.900000, 0.933333, 1]
    
    
    ################################################# handle type 1 columns
    type_one_column_list = [
       'question_conversational', \
    #    'question_expect_short_answer', \
    #    'question_fact_seeking', \
       'question_has_commonly_accepted_answer', \
    #    'question_multi_intent', \
       'question_not_really_a_question', \
       'question_opinion_seeking', \
       'question_type_choice', \
       'question_type_compare', \
       'question_type_consequence', \
       'question_type_definition', \
       'question_type_entity', \
       'question_type_instructions', \
    #    'question_type_procedure', \
    #    'question_type_reason_explanation', \
    #    'answer_type_instructions'
    #    'answer_type_procedure', \
    #    'answer_type_reason_explanation', \
    ]
    
    oof_df[type_one_column_list] = scaler.fit_transform(oof_df[type_one_column_list])
    
    for column in type_one_column_list:
        
        oof_df.loc[oof_df[column] <= 0.16667, column] = -0
        oof_df.loc[(oof_df[column] > 0.16667) & (oof_df[column] <= 0.41667), column] = -0.333333
        oof_df.loc[(oof_df[column] > 0.41667) & (oof_df[column] <= 0.58333), column] = -0.500000
        oof_df.loc[(oof_df[column] > 0.58333) & (oof_df[column] <= 0.73333), column] = -0.666667
        oof_df.loc[(oof_df[column] > 0.73333), column] = -1
        
        oof_df.loc[:, column] = -1 * oof_df.loc[:, column]
    
    ################################################# handle type 2 columns      
    type_two_column_list = [
        'question_type_spelling'
    ]
    
    oof_df[type_two_column_list] = scaler.fit_transform(oof_df[type_two_column_list])
    
    for column in type_two_column_list:
        oof_df.loc[oof_df[column] <= 0.16667, column] = -0
        oof_df.loc[(oof_df[column] > 0.16667) & (oof_df[column] <= 0.5), column] = -0.333333
        oof_df.loc[(oof_df[column] > 0.5), column] = -0.666667
        
        oof_df.loc[:, column] = -1 * oof_df.loc[:, column]
    
    
    
    ################################################# handle type 3 columns      
    type_three_column_list = [
    #    'question_asker_intent_understanding', \
    #    'question_body_critical', \
    #    'question_interestingness_others', \
       'question_interestingness_self', \
    #    'question_well_written', \
    #    'answer_helpful', \
    #    'answer_level_of_information', \
    #    'answer_plausible', \
    #    'answer_relevance', \
    #    'answer_well_written'
    ]

    oof_df[type_three_column_list] = scaler.fit_transform(oof_df[type_three_column_list])
    
    for column in type_three_column_list:
        oof_df.loc[oof_df[column] <= 0.388889, column] = -0.333333
        oof_df.loc[(oof_df[column] > 0.388889) & (oof_df[column] <= 0.472222), column] = -0.444444
        oof_df.loc[(oof_df[column] > 0.472222) & (oof_df[column] <= 0.527778), column] = -0.5
        oof_df.loc[(oof_df[column] > 0.527778) & (oof_df[column] <= 0.611111), column] = -0.555556
        oof_df.loc[(oof_df[column] > 0.611111) & (oof_df[column] <= 0.722222), column] = -0.666667
        oof_df.loc[(oof_df[column] > 0.722222) & (oof_df[column] <= 0.805556), column] = -0.777778
        oof_df.loc[(oof_df[column] > 0.805556) & (oof_df[column] <= 0.861111), column] = -0.833333
        oof_df.loc[(oof_df[column] > 0.861111) & (oof_df[column] <= 0.944445), column] = -0.888889
        oof_df.loc[(oof_df[column] > 0.944445), column] = -1
        
        oof_df.loc[:, column] = -1 * oof_df.loc[:, column]
        
        
    ################################################# handle type 4 columns      
    type_four_column_list = [
        'answer_satisfaction'
    ]
    
    oof_df[type_four_column_list] = scaler.fit_transform(oof_df[type_four_column_list])
    
    for column in type_four_column_list:
        
        oof_df.loc[oof_df[column] <= 0.233333, column] = -0.200000
        oof_df.loc[(oof_df[column] > 0.233333) & (oof_df[column] <= 0.283333), column] = -0.266667
        oof_df.loc[(oof_df[column] > 0.283333) & (oof_df[column] <= 0.316666), column] = -0.300000
        oof_df.loc[(oof_df[column] > 0.316666) & (oof_df[column] <= 0.366666), column] = -0.333333
        oof_df.loc[(oof_df[column] > 0.366666) & (oof_df[column] <= 0.433333), column] = -0.400000
        oof_df.loc[(oof_df[column] > 0.433333) & (oof_df[column] <= 0.483333), column] = -0.466667
        oof_df.loc[(oof_df[column] > 0.483333) & (oof_df[column] <= 0.516666), column] = -0.500000
        oof_df.loc[(oof_df[column] > 0.516666) & (oof_df[column] <= 0.566666), column] = -0.533333
        oof_df.loc[(oof_df[column] > 0.566666) & (oof_df[column] <= 0.633333), column] = -0.600000
        oof_df.loc[(oof_df[column] > 0.633333) & (oof_df[column] <= 0.683333), column] = -0.666667
        oof_df.loc[(oof_df[column] > 0.683333) & (oof_df[column] <= 0.716666), column] = -0.700000
        oof_df.loc[(oof_df[column] > 0.716666) & (oof_df[column] <= 0.766666), column] = -0.733333
        oof_df.loc[(oof_df[column] > 0.767666) & (oof_df[column] <= 0.833333), column] = -0.800000
        oof_df.loc[(oof_df[column] > 0.833333) & (oof_df[column] <= 0.883333), column] = -0.866667
        oof_df.loc[(oof_df[column] > 0.883333) & (oof_df[column] <= 0.916666), column] = -0.900000
        oof_df.loc[(oof_df[column] > 0.916666) & (oof_df[column] <= 0.966666), column] = -0.933333
        oof_df.loc[(oof_df[column] > 0.966666), column] = -1
        
        oof_df.loc[:, column] = -1 * oof_df.loc[:, column]
    
    
    ################################################# round to i / 90 (i from 0 to 90)
    oof_values = oof_df[target_columns].values
    oof_values = np.around(oof_values * 90) / 90
    oof_df[target_columns] = oof_values
    
    return oof_df

def postprocessing_v2(oof_df):
       
    scaler = MinMaxScaler()
    
    # type 1 column [0, 0.333333, 0.5, 0.666667, 1]
    # type 2 column [0, 0.333333, 0.666667]
    # type 3 column [0.333333, 0.444444, 0.5, 0.555556, 0.666667, 0.777778, 0.8333333, 0.888889, 1]
    # type 4 column [0.200000, 0.266667, 0.300000, 0.333333, 0.400000, \
    # 0.466667, 0.5, 0.533333, 0.600000, 0.666667, 0.700000, \
    # 0.733333, 0.800000, 0.866667, 0.900000, 0.933333, 1]
    
    # comment some columns based on oof result
    
    ################################################# handle type 1 columns
    type_one_column_list = [
       'question_conversational', \
       'question_has_commonly_accepted_answer', \
       'question_not_really_a_question', \
       'question_type_choice', \
       'question_type_compare', \
       'question_type_consequence', \
       'question_type_definition', \
       'question_type_entity', \
       'question_type_instructions', 
    ]
    
    oof_df[type_one_column_list] = scaler.fit_transform(oof_df[type_one_column_list])
    
    tmp = oof_df.copy(deep=True)
    
    for column in type_one_column_list:
        
        oof_df.loc[tmp[column] <= 0.16667, column] = 0
        oof_df.loc[(tmp[column] > 0.16667) & (tmp[column] <= 0.41667), column] = 0.333333
        oof_df.loc[(tmp[column] > 0.41667) & (tmp[column] <= 0.58333), column] = 0.500000
        oof_df.loc[(tmp[column] > 0.58333) & (tmp[column] <= 0.73333), column] = 0.666667
        oof_df.loc[(tmp[column] > 0.73333), column] = 1
    
    
    
    ################################################# handle type 2 columns      
#     type_two_column_list = [
#         'question_type_spelling'
#     ]
    
#     for column in type_two_column_list:
#         if sum(tmp[column] > 0.15)>0:
#             oof_df.loc[tmp[column] <= 0.15, column] = 0
#             oof_df.loc[(tmp[column] > 0.15) & (tmp[column] <= 0.45), column] = 0.333333
#             oof_df.loc[(tmp[column] > 0.45), column] = 0.666667
#         else:
#             t1 = max(int(len(tmp[column])*0.0013),2)
#             t2 = max(int(len(tmp[column])*0.0008),1)
#             thred1 = sorted(list(tmp[column]))[-t1]
#             thred2 = sorted(list(tmp[column]))[-t2]
#             oof_df.loc[tmp[column] <= thred1, column] = 0
#             oof_df.loc[(tmp[column] > thred1) & (tmp[column] <= thred2), column] = 0.333333
#             oof_df.loc[(tmp[column] > thred2), column] = 0.666667
    
    
    
    ################################################# handle type 3 columns      
    type_three_column_list = [
       'question_interestingness_self', 
    ]
    scaler = MinMaxScaler(feature_range=(0, 1))
    oof_df[type_three_column_list] = scaler.fit_transform(oof_df[type_three_column_list])
    tmp[type_three_column_list] = scaler.fit_transform(tmp[type_three_column_list])
    
    for column in type_three_column_list:
        oof_df.loc[tmp[column] <= 0.385, column] = 0.333333
        oof_df.loc[(tmp[column] > 0.385) & (tmp[column] <= 0.47), column] = 0.444444
        oof_df.loc[(tmp[column] > 0.47) & (tmp[column] <= 0.525), column] = 0.5
        oof_df.loc[(tmp[column] > 0.525) & (tmp[column] <= 0.605), column] = 0.555556
        oof_df.loc[(tmp[column] > 0.605) & (tmp[column] <= 0.715), column] = 0.666667
        oof_df.loc[(tmp[column] > 0.715) & (tmp[column] <= 0.8), column] = 0.833333
        oof_df.loc[(tmp[column] > 0.8) & (tmp[column] <= 0.94), column] = 0.888889
        oof_df.loc[(tmp[column] > 0.94), column] = 1
        
        
        
    ################################################# handle type 4 columns      
    type_four_column_list = [
        'answer_satisfaction'
    ]
    scaler = MinMaxScaler(feature_range=(0.2, 1))
    oof_df[type_four_column_list] = scaler.fit_transform(oof_df[type_four_column_list])
    tmp[type_four_column_list] = scaler.fit_transform(tmp[type_four_column_list])
    
    for column in type_four_column_list:
        
        oof_df.loc[tmp[column] <= 0.233, column] = 0.200000
        oof_df.loc[(tmp[column] > 0.233) & (tmp[column] <= 0.283), column] = 0.266667
        oof_df.loc[(tmp[column] > 0.283) & (tmp[column] <= 0.315), column] = 0.300000
        oof_df.loc[(tmp[column] > 0.315) & (tmp[column] <= 0.365), column] = 0.333333
        oof_df.loc[(tmp[column] > 0.365) & (tmp[column] <= 0.433), column] = 0.400000
        oof_df.loc[(tmp[column] > 0.433) & (tmp[column] <= 0.483), column] = 0.466667
        oof_df.loc[(tmp[column] > 0.483) & (tmp[column] <= 0.517), column] = 0.500000
        oof_df.loc[(tmp[column] > 0.517) & (tmp[column] <= 0.567), column] = 0.533333
        oof_df.loc[(tmp[column] > 0.567) & (tmp[column] <= 0.633), column] = 0.600000
        oof_df.loc[(tmp[column] > 0.633) & (tmp[column] <= 0.683), column] = 0.666667
        oof_df.loc[(tmp[column] > 0.683) & (tmp[column] <= 0.715), column] = 0.700000
        oof_df.loc[(tmp[column] > 0.715) & (tmp[column] <= 0.767), column] = 0.733333
        oof_df.loc[(tmp[column] > 0.767) & (tmp[column] <= 0.833), column] = 0.800000
        oof_df.loc[(tmp[column] > 0.883) & (tmp[column] <= 0.915), column] = 0.900000
        oof_df.loc[(tmp[column] > 0.915) & (tmp[column] <= 0.967), column] = 0.933333
        oof_df.loc[(tmp[column] > 0.967), column] = 1
    
    
    ################################################# round to i / 90 (i from 0 to 90)
    oof_values = oof_df[TARGET_COLUMNS].values
    DEGREE = len(oof_df)//45*9
    # if degree:
    #     DEGREE = degree
    # DEGREE = 90
    oof_values = np.around(oof_values * DEGREE) / DEGREE  ### 90 To be changed
    oof_df[TARGET_COLUMNS] = oof_values
    
    return oof_df


if __name__ == "__main__":
    
    model_folder = "/media/jionie/my_disk/Kaggle/Google_Quest_Answer/model/"
    oof_bert_base_cased = pd.read_csv(model_folder + "oof_bert_base_cased_swa.csv")
    oof_bert_base_cased_two_model = pd.read_csv(model_folder + "oof_bert_base_cased_two_model_swa.csv")
    oof_bert_base_uncased = pd.read_csv(model_folder + "oof_bert_base_uncased_swa.csv")
    oof_bert_base_uncased_two_model = pd.read_csv(model_folder + "oof_bert_base_uncased_two_model_swa.csv")
    oof_xlnet_base_cased = pd.read_csv(model_folder + "oof_xlnet_base_cased.csv")
    oof_xlnet_base_cased_two_model = pd.read_csv(model_folder + "oof_xlnet_base_cased_two_model_swa.csv")

    # oof_df = (oof_bert_base_cased[TARGET_COLUMNS] + oof_bert_base_cased_two_model[TARGET_COLUMNS] + \
    #           oof_bert_base_uncased[TARGET_COLUMNS] + oof_bert_base_uncased_two_model[TARGET_COLUMNS] + \
    #           oof_xlnet_base_cased[TARGET_COLUMNS])/5.0
    
    oof_df = (oof_bert_base_cased[TARGET_COLUMNS] + oof_bert_base_cased_two_model[TARGET_COLUMNS] + \
              oof_bert_base_uncased[TARGET_COLUMNS] + oof_bert_base_uncased_two_model[TARGET_COLUMNS] + \
              oof_xlnet_base_cased[TARGET_COLUMNS] + oof_xlnet_base_cased_two_model[TARGET_COLUMNS])/6.0
    
    # oof_two_model_bert = (oof_bert_base_cased_two_model[TARGET_COLUMNS] + oof_bert_base_uncased_two_model[TARGET_COLUMNS]) / 2.0
    # oof_two_model = (oof_two_model_bert[TARGET_COLUMNS] + oof_xlnet_base_cased_two_model[TARGET_COLUMNS]) / 2.0
    
    # oof_bert = (oof_bert_base_cased[TARGET_COLUMNS] + oof_bert_base_uncased[TARGET_COLUMNS]) / 2.0
    # oof_single_model = (oof_bert[TARGET_COLUMNS] + oof_xlnet_base_cased[TARGET_COLUMNS]) / 2.0
    
    # oof_df = (oof_two_model + oof_single_model) / 2.0
    
    # oof_df = ((oof_bert_base_cased[TARGET_COLUMNS] +  oof_bert_base_uncased[TARGET_COLUMNS] + \
    #           (oof_bert_base_cased_two_model[TARGET_COLUMNS] + oof_bert_base_uncased_two_model[TARGET_COLUMNS])/2)/3 + \
    #           (oof_xlnet_base_cased[TARGET_COLUMNS] + oof_xlnet_base_cased_two_model[TARGET_COLUMNS])/2)/2.0

    train_df = pd.read_csv("/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/train.csv")
    
    # oof_df = postprocessing(oof_df, TARGET_COLUMNS)
    oof_df = postprocessing_v2(oof_df)
    
    spearman = Spearman(train_df[TARGET_COLUMNS].values, oof_df[TARGET_COLUMNS].values)
    
    print(spearman)