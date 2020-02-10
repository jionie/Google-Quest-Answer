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
from scipy.stats import rankdata

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

# question_not_really_a_question
# question_type_definition
# question_type_spelling

def postprocess_special_columns(oof_df, special_columns):
    
    scaler = MinMaxScaler()
    oof_df[special_columns] = scaler.fit_transform(oof_df[special_columns])
    
    for column in special_columns:
        
        if column == "question_not_really_a_question":
            oof_df.loc[(oof_df[column] <= 0.16667) & (oof_df["category"]=="LIFE_ARTS"), column] = -0
            oof_df.loc[(oof_df[column] > 0.16667) & (oof_df["category"]=="LIFE_ARTS"), column] = -0.333333
            
            oof_df.loc[(oof_df[column] <= 0.16667) & (oof_df["category"]=="CULTURE"), column] = -0
            oof_df.loc[(oof_df[column] > 0.16667) & (oof_df[column] <= 0.41667) & (oof_df["category"]=="CULTURE"), column] = -0.333333
            oof_df.loc[(oof_df[column] > 0.41667) & (oof_df[column] <= 0.58333) & (oof_df["category"]=="CULTURE"), column] = -0.500000
            oof_df.loc[(oof_df[column] > 0.58333) & (oof_df["category"]=="CULTURE"), column] = -0.666667
            
            oof_df.loc[(oof_df[column] <= 0.16667) & (oof_df["category"]=="SCIENCE"), column] = -0
            oof_df.loc[(oof_df[column] > 0.16667) & (oof_df[column] <= 0.41667) & (oof_df["category"]=="SCIENCE"), column] = -0.333333
            oof_df.loc[(oof_df[column] > 0.41667) & (oof_df["category"]=="SCIENCE"), column] = -0.500000
            
            oof_df.loc[(oof_df[column] <= 0.16667) & (oof_df["category"]=="STACKOVERFLOW"), column] = -0
            oof_df.loc[(oof_df[column] > 0.16667) & (oof_df[column] <= 0.41667) & (oof_df["category"]=="STACKOVERFLOW"), column] = -0.333333
            oof_df.loc[(oof_df[column] > 0.41667) & (oof_df["category"]=="STACKOVERFLOW"), column] = -0.500000
            
            oof_df.loc[(oof_df[column] <= 0.16667) & (oof_df["category"]=="TECHNOLOGY"), column] = -0
            oof_df.loc[(oof_df[column] > 0.16667) & (oof_df[column] <= 0.41667) & (oof_df["category"]=="TECHNOLOGY"), column] = -0.333333
            oof_df.loc[(oof_df[column] > 0.41667) & (oof_df[column] <= 0.58333) & (oof_df["category"]=="TECHNOLOGY"), column] = -0.500000
            oof_df.loc[(oof_df[column] > 0.58333) & (oof_df[column] <= 0.83333) & (oof_df["category"]=="TECHNOLOGY"), column] = -0.666667
            oof_df.loc[(oof_df[column] > 0.83333) & (oof_df["category"]=="TECHNOLOGY"), column] = -1
            
            oof_df.loc[:, column] = -1 * oof_df.loc[:, column]
            
        if column == "question_type_definition":
            
            oof_df.loc[(oof_df[column] <= 0.16667) & (oof_df["category"]=="LIFE_ARTS"), column] = -0
            oof_df.loc[(oof_df[column] > 0.16667) & (oof_df[column] <= 0.5) & (oof_df["category"]=="LIFE_ARTS"), column] = -0.333333
            oof_df.loc[(oof_df[column] > 0.5) & (oof_df[column] <= 0.83333) & (oof_df["category"]=="LIFE_ARTS"), column] = -0.666667
            oof_df.loc[(oof_df[column] > 0.83333) & (oof_df["category"]=="LIFE_ARTS"), column] = -1
            
            oof_df.loc[(oof_df[column] <= 0.16667) & (oof_df["category"]=="CULTURE"), column] = -0
            oof_df.loc[(oof_df[column] > 0.16667) & (oof_df[column] <= 0.41667) & (oof_df["category"]=="CULTURE"), column] = -0.333333
            oof_df.loc[(oof_df[column] > 0.41667) & (oof_df[column] <= 0.58333) & (oof_df["category"]=="CULTURE"), column] = -0.500000
            oof_df.loc[(oof_df[column] > 0.58333) & (oof_df[column] <= 0.83333) & (oof_df["category"]=="CULTURE"), column] = -0.666667
            oof_df.loc[(oof_df[column] > 0.83333) & (oof_df["category"]=="CULTURE"), column] = -1
            
            oof_df.loc[(oof_df[column] <= 0.16667) & (oof_df["category"]=="SCIENCE"), column] = -0
            oof_df.loc[(oof_df[column] > 0.16667) & (oof_df[column] <= 0.41667) & (oof_df["category"]=="SCIENCE"), column] = -0.333333
            oof_df.loc[(oof_df[column] > 0.41667) & (oof_df[column] <= 0.58333) & (oof_df["category"]=="SCIENCE"), column] = -0.500000
            oof_df.loc[(oof_df[column] > 0.58333) & (oof_df[column] <= 0.83333) & (oof_df["category"]=="SCIENCE"), column] = -0.666667
            oof_df.loc[(oof_df[column] > 0.83333) & (oof_df["category"]=="SCIENCE"), column] = -1
            
            oof_df.loc[(oof_df[column] <= 0.16667) & (oof_df["category"]=="STACKOVERFLOW"), column] = -0
            oof_df.loc[(oof_df[column] > 0.16667) & (oof_df[column] <= 0.41667) & (oof_df["category"]=="STACKOVERFLOW"), column] = -0.333333
            oof_df.loc[(oof_df[column] > 0.41667) & (oof_df[column] <= 0.58333) & (oof_df["category"]=="STACKOVERFLOW"), column] = -0.500000
            oof_df.loc[(oof_df[column] > 0.58333) & (oof_df[column] <= 0.83333) & (oof_df["category"]=="STACKOVERFLOW"), column] = -0.666667
            oof_df.loc[(oof_df[column] > 0.83333) & (oof_df["category"]=="STACKOVERFLOW"), column] = -1
            
            oof_df.loc[(oof_df[column] <= 0.16667) & (oof_df["category"]=="TECHNOLOGY"), column] = -0
            oof_df.loc[(oof_df[column] > 0.16667) & (oof_df[column] <= 0.41667) & (oof_df["category"]=="TECHNOLOGY"), column] = -0.333333
            oof_df.loc[(oof_df[column] > 0.41667) & (oof_df[column] <= 0.58333) & (oof_df["category"]=="TECHNOLOGY"), column] = -0.500000
            oof_df.loc[(oof_df[column] > 0.58333) & (oof_df[column] <= 0.83333) & (oof_df["category"]=="TECHNOLOGY"), column] = -0.666667
            oof_df.loc[(oof_df[column] > 0.83333) & (oof_df["category"]=="TECHNOLOGY"), column] = -1
            
            oof_df.loc[:, column] = -1 * oof_df.loc[:, column]
            
        if column == "question_type_spelling":
            
            oof_df.loc[oof_df["category"]=="LIFE_ARTS", column] = -0
            
            oof_df.loc[(oof_df[column] <= 0.16667) & (oof_df["category"]=="CULTURE"), column] = -0
            oof_df.loc[(oof_df[column] > 0.16667) & (oof_df[column] <= 0.5) & (oof_df["category"]=="CULTURE"), column] = -0.333333
            oof_df.loc[(oof_df[column] > 0.5) & (oof_df["category"]=="CULTURE"), column] = -0.666667
            
            oof_df.loc[oof_df["category"]=="SCIENCE", column] = -0
            oof_df.loc[oof_df["category"]=="STACKOVERFLOW", column] = -0
            oof_df.loc[oof_df["category"]=="TECHNOLOGY", column] = -0
            
            oof_df.loc[:, column] = -1 * oof_df.loc[:, column]
       
    return oof_df


def rank_average(preds):
    cols = range(30)
    preds_avg = np.empty_like(preds[0])
    for col in cols:
        avg_rank = np.mean([rankdata(p[:, col], method='dense') for p in preds], axis=0).round(0).astype(int)
        avg_rank = rankdata(avg_rank, method='dense')-1
        nunique = np.unique(avg_rank).shape[0]
        arrays = np.array([p[:,col] for p in preds])
        ranges = arrays.min(), arrays.max()
        uniform = np.linspace(ranges[0],ranges[1],nunique)
        refined = np.array([uniform[i] for i in avg_rank])
        preds_avg[:,col] = refined
    return preds_avg

            

def postprocessing(oof_df, target_columns):
    
    scaler = MinMaxScaler()
    tmp = oof_df.copy(deep=True)
    
    ################################################# handle type 1 columns
    type_one_column_list = [
       'question_conversational', \
       'question_has_commonly_accepted_answer', \
       'question_not_really_a_question', \
       'question_opinion_seeking', \
       'question_type_choice', \
       'question_type_compare', \
       'question_type_consequence', \
       'question_type_definition', \
       'question_type_entity', \
       'question_type_instructions'
    ]
    
    type_one_column_norm_list = [
       'question_conversational', \
       'question_has_commonly_accepted_answer', \
       'question_not_really_a_question', \
       'question_opinion_seeking', \
       'question_type_choice', \
       'question_type_compare', \
       'question_type_consequence', \
       'question_type_entity', \
       'question_type_instructions'
    ]
    
    
    oof_df[type_one_column_norm_list] = scaler.fit_transform(oof_df[type_one_column_norm_list])
    tmp[type_one_column_norm_list] = scaler.fit_transform(tmp[type_one_column_norm_list])
    
    for column in type_one_column_list:
        
        oof_df.loc[tmp[column] <= ((0.333333 + 0)/2), column] = 0
        oof_df.loc[(tmp[column] > ((0.333333 + 0)/2)) & (tmp[column] <= ((0.500000 + 0.333333)/2)), column] = 0.333333
        oof_df.loc[(tmp[column] > ((0.500000 + 0.333333)/2)) & (tmp[column] <= ((0.666667 + 0.500000)/2)), column] = 0.500000
        oof_df.loc[(tmp[column] > ((0.666667 + 0.500000)/2)) & (tmp[column] <= ((1 + 0.666667)/2)), column] = 0.666667
        oof_df.loc[(tmp[column] > ((1 + 0.666667)/2)), column] = 1
    
    
    
    ################################################# handle type 2 columns      
    type_two_column_list = [
        'question_type_spelling'
    ]
    
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # oof_df[type_two_column_list] = scaler.fit_transform(oof_df[type_two_column_list])
    # tmp[type_two_column_list] = scaler.fit_transform(tmp[type_two_column_list])
    
    # for column in type_two_column_list:
   
    #     oof_df.loc[tmp[column] <= ((0.333333 + 0)/2), column] = 0
    #     oof_df.loc[(tmp[column] > ((0.333333 + 0)/2)) & (tmp[column] <= ((0.666667 + 0.333333)/2)), column] = 0.333333
    #     oof_df.loc[(tmp[column] > ((0.666667 + 0.333333)/2)), column] = 0.666667
        
    # for column in type_two_column_list:
    #     if sum(tmp[column] > 0.15)>0:
    #         oof_df.loc[tmp[column] <= ((0.333333 + 0)/2), column] = 0
    #         oof_df.loc[(tmp[column] > ((0.333333 + 0)/2)) & (tmp[column] <= ((0.666667 + 0.333333)/2)), column] = 0.333333
    #         oof_df.loc[(tmp[column] > ((0.666667 + 0.333333)/2)), column] = 0.666667
    #     else:
    #         t1 = max(int(len(tmp[column])*0.0013),2)
    #         t2 = max(int(len(tmp[column])*0.0008),1)
    #         thred1 = sorted(list(tmp[column]))[-t1]
    #         thred2 = sorted(list(tmp[column]))[-t2]
    #         oof_df.loc[tmp[column] <= thred1, column] = 0
    #         oof_df.loc[(tmp[column] > thred1) & (tmp[column] <= thred2), column] = 0.333333
    #         oof_df.loc[(tmp[column] > thred2), column] = 0.666667

    
    
    ################################################# handle type 3 columns      
    type_three_column_list = [
       'question_interestingness_self', \
    ]
    
    type_one_column_norm_list = [
       'question_interestingness_self', \
    ]
    
    scaler = MinMaxScaler(feature_range=(0.333333, 1))
    oof_df[type_one_column_norm_list] = scaler.fit_transform(oof_df[type_one_column_norm_list])
    tmp[type_one_column_norm_list] = scaler.fit_transform(tmp[type_one_column_norm_list])
    
    for column in type_three_column_list:
        oof_df.loc[tmp[column] <=  ((0.444444 + 0.333333)/2), column] = 0.333333
        oof_df.loc[(tmp[column] > ((0.444444 + 0.333333)/2)) & (tmp[column] <= ((0.5 + 0.444444)/2)), column] = 0.444444
        oof_df.loc[(tmp[column] > ((0.5 + 0.444444)/2)) & (tmp[column] <= ((0.555556 + 0.5)/2)), column] = 0.5
        oof_df.loc[(tmp[column] > ((0.555556 + 0.5)/2)) & (tmp[column] <= ((0.666667 + 0.555556)/2)), column] = 0.555556
        oof_df.loc[(tmp[column] > ((0.666667 + 0.555556)/2)) & (tmp[column] <= ((0.833333 + 0.666667)/2)), column] = 0.666667
        oof_df.loc[(tmp[column] > ((0.833333 + 0.666667)/2)) & (tmp[column] <= ((0.888889 + 0.833333)/2)), column] = 0.833333
        oof_df.loc[(tmp[column] > ((0.888889 + 0.833333)/2)) & (tmp[column] <= ((1 + 0.888889)/2)), column] = 0.888889
        oof_df.loc[(tmp[column] > ((1 + 0.888889)/2)), column] = 1
        
        
        
    ################################################# handle type 4 columns      
    type_four_column_list = [
        'answer_satisfaction'
    ]
    scaler = MinMaxScaler(feature_range=(0, 1))
    oof_df[type_four_column_list] = scaler.fit_transform(oof_df[type_four_column_list])
    tmp[type_four_column_list] = scaler.fit_transform(tmp[type_four_column_list])
    
    for column in type_four_column_list:
        
        oof_df.loc[tmp[column] <= ((0.266667 + 0.200000)/2), column] = 0.200000
        oof_df.loc[(tmp[column] > ((0.266667 + 0.200000)/2)) & (tmp[column] <= ((0.300000 + 0.266667)/2)), column] = 0.266667
        oof_df.loc[(tmp[column] > ((0.300000 + 0.266667)/2)) & (tmp[column] <= ((0.333333 + 0.300000)/2)), column] = 0.300000
        oof_df.loc[(tmp[column] > ((0.333333 + 0.300000)/2)) & (tmp[column] <= ((0.400000 + 0.333333)/2)), column] = 0.333333
        oof_df.loc[(tmp[column] > ((0.400000 + 0.333333)/2)) & (tmp[column] <= ((0.466667 + 0.400000)/2)), column] = 0.400000
        oof_df.loc[(tmp[column] > ((0.466667 + 0.400000)/2)) & (tmp[column] <= ((0.500000 + 0.466667)/2)), column] = 0.466667
        oof_df.loc[(tmp[column] > ((0.500000 + 0.466667)/2)) & (tmp[column] <= ((0.533333 + 0.500000)/2)), column] = 0.500000
        oof_df.loc[(tmp[column] > ((0.533333 + 0.500000)/2)) & (tmp[column] <= ((0.600000 + 0.533333)/2)), column] = 0.533333
        oof_df.loc[(tmp[column] > ((0.600000 + 0.533333)/2)) & (tmp[column] <= ((0.666667 + 0.600000)/2)), column] = 0.600000
        oof_df.loc[(tmp[column] > ((0.666667 + 0.600000)/2)) & (tmp[column] <= ((0.700000 + 0.666667)/2)), column] = 0.666667
        oof_df.loc[(tmp[column] > ((0.700000 + 0.666667)/2)) & (tmp[column] <= ((0.733333 + 0.700000)/2)), column] = 0.700000
        oof_df.loc[(tmp[column] > ((0.733333 + 0.700000)/2)) & (tmp[column] <= ((0.800000 + 0.733333)/2)), column] = 0.733333
        oof_df.loc[(tmp[column] > ((0.800000 + 0.733333)/2)) & (tmp[column] <= ((0.900000 + 0.800000)/2)), column] = 0.800000
        oof_df.loc[(tmp[column] > ((0.900000 + 0.800000)/2)) & (tmp[column] <= ((0.933333 + 0.900000)/2)), column] = 0.900000
        oof_df.loc[(tmp[column] > ((0.933333 + 0.900000)/2)) & (tmp[column] <= ((1 + 0.933333)/2)), column] = 0.933333
        oof_df.loc[(tmp[column] > ((1 + 0.933333)/2)), column] = 1
    
    
    ################################################# round to i / 90 (i from 0 to 90)
    oof_values = oof_df[TARGET_COLUMNS].values
    DEGREE = len(oof_df)//45*9

    oof_values = np.around(oof_values * DEGREE) / DEGREE  ### 90 To be changed
    oof_df[TARGET_COLUMNS] = oof_values
    
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
    type_two_column_list = [
        'question_type_spelling'
    ]
    
    for column in type_two_column_list:
        if sum(tmp[column] > 0.15)>0:
            oof_df.loc[tmp[column] <= 0.15, column] = 0
            oof_df.loc[(tmp[column] > 0.15) & (tmp[column] <= 0.45), column] = 0.333333
            oof_df.loc[(tmp[column] > 0.45), column] = 0.666667
        else:
            t1 = max(int(len(tmp[column])*0.0013),2)
            t2 = max(int(len(tmp[column])*0.0008),1)
            thred1 = sorted(list(tmp[column]))[-t1]
            thred2 = sorted(list(tmp[column]))[-t2]
            oof_df.loc[tmp[column] <= thred1, column] = 0
            oof_df.loc[(tmp[column] > thred1) & (tmp[column] <= thred2), column] = 0.333333
            oof_df.loc[(tmp[column] > thred2), column] = 0.666667
    
    
    
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
#     if degree:
#         DEGREE = degree
#     DEGREE = 90
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
    oof_roberta_base_two_model = pd.read_csv(model_folder + "oof_roberta_base_two_model_swa.csv")
    oof_roberta_base = pd.read_csv(model_folder + "oof_roberta_base_swa.csv")

    
    oof_df = 0.25 * (0.5 * oof_bert_base_cased[TARGET_COLUMNS] + 0.5 * oof_bert_base_cased_two_model[TARGET_COLUMNS]) + \
                0.25 * (0.5 * oof_bert_base_uncased[TARGET_COLUMNS] + 0.5 * oof_bert_base_uncased_two_model[TARGET_COLUMNS]) + \
                0.25 * (0.5 * oof_xlnet_base_cased[TARGET_COLUMNS] + 0.5 * oof_xlnet_base_cased_two_model[TARGET_COLUMNS]) + \
                0.25 * (0.5 * oof_roberta_base[TARGET_COLUMNS] + 0.5 * oof_roberta_base_two_model[TARGET_COLUMNS])   
              
    


    train_df = pd.read_csv("/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/train.csv")
    oof_df["category"] = train_df["category"]
    
    # oof_df[TARGET_COLUMNS] = rank_average(oof_df[TARGET_COLUMNS].values)
    
    oof_df = postprocessing(oof_df, TARGET_COLUMNS)
    # oof_df = postprocessing_v2(oof_df)

    # n = train_df['url'].apply(lambda x:(('ell.stackexchange.com' in x) or ('english.stackexchange.com' in x))).tolist()

    # for idx, x in enumerate(n):
    #     if x:
    #         oof_df.loc[idx, 'question_type_spelling'] = 0.5
            
    # oof_df.loc[\
    #     (train_df.category == "CULTURE") & ((train_df.host == "english.stackexchange.com") | (train_df.host == "ell.stackexchange.com")), \
    #         'question_type_spelling'] = 1
    
    vocab_list_large = [
    'pronounced', 'pronounce', 'pronunciation', 'adjective', 'syllables', 'spell', 'sounds', 'Ngram', 'verb'
    ]
    def rule_large(x):
        if x < 3:
            return 0.0
        elif x < 6:
            return 1/3
        else:
            return 2/3
        
    y_preds_question_type_spelling = (train_df['question_body'].apply(lambda x: sum([x.count(vocab) for vocab in vocab_list_large]))\
        ).apply(rule_large)
            
    
    y_preds_question_type_spelling_copy = list(y_preds_question_type_spelling)
    
    n = train_df['url'].apply(lambda x:(('ell.stackexchange.com' in x) or ('english.stackexchange.com' in x))).tolist()
    spelling=[]
    for x in n:
        if x:
            spelling.append(1/3)
        else:
            spelling.append(0.)
    
    oof_df['question_type_spelling'] = spelling
       
    # if (len(set(y_preds_question_type_spelling)) == 1):
    #     oof_df['question_type_spelling'] = spelling
    # else:
    #     oof_df['question_type_spelling'] = y_preds_question_type_spelling
        
    # for i in range(len(spelling)):
    #     if (spelling[i] != 0):
    #         oof_df.loc[i, 'question_type_spelling'] = spelling[i]
    #         break
        
    # for i in range(len(spelling)-1, 0, -1):
    #     if (spelling[i] != 0):
    #         oof_df.loc[i, 'question_type_spelling'] = spelling[i] * 2
    #         break
            
    # vocab_list_base = [
    #     'sound', 'prefix', 'adjective', 'verb', 'noun', 'word', 'Ngram', 'conversation', 'syllable'
    # ]
    # def rule_base(x):
    #     if x == 0:
    #         return 0.0
    #     elif x == 1:
    #         return 1/64
    #     else:
    #         return 1/32
        
    # y_preds_question_type_spelling = (
    #     train_df['question_title'].apply(
    #         lambda x: sum([x.count(vocab) for vocab in vocab_list_large])
    #     ) + train_df['question_body'].apply(
    #         lambda x: sum([x.count(vocab) for vocab in vocab_list_large])
    #     )).apply(rule_large) + (train_df['question_title'].apply(
    #         lambda x: sum([x.count(vocab) for vocab in vocab_list_base])
    #     ) + train_df['question_body'].apply(
    #         lambda x: sum([x.count(vocab) for vocab in vocab_list_base])
    #     )).apply(rule_base)
    
    print(oof_df['question_type_spelling'].value_counts())
    
    # n = train_df['url'].apply(lambda x:(('ell.stackexchange.com' in x) or ('english.stackexchange.com' in x))).tolist()

    # for idx, x in enumerate(n):
    #     if x:
    #         if (oof_df.loc[idx, 'category'] == 'CULTURE'):
    #             oof_df.loc[idx, 'question_type_spelling'] = 1/3
    
    spearman, spearman_list = Spearman(train_df[TARGET_COLUMNS].values, oof_df[TARGET_COLUMNS].values)
    
    for i in range(len(TARGET_COLUMNS)):
        print(TARGET_COLUMNS[i], ":", spearman_list[i])
    
    print(spearman)
    
    
    
# question_asker_intent_understanding : 0.3913240782174862
# question_body_critical : 0.6505225400884914
# question_conversational : 0.49628898642029673
# question_expect_short_answer : 0.32475526913414987
# question_fact_seeking : 0.3817788468272073
# question_has_commonly_accepted_answer : 0.47230037109903056
# question_interestingness_others : 0.36998401526413244
# question_interestingness_self : 0.5113239919964795
# question_multi_intent : 0.6113465378804653
# question_not_really_a_question : 0.1289263771516838
# question_opinion_seeking : 0.47941525675967983
# question_type_choice : 0.7692508885743983
# question_type_compare : 0.5456676906025371
# question_type_consequence : 0.25940544311123
# question_type_definition : 0.6240250166600889
# question_type_entity : 0.6106638880179526
# question_type_instructions : 0.7994884073149504
# question_type_procedure : 0.37509398018350876
# question_type_reason_explanation : 0.694063016795095
# question_type_spelling : 0.14111060200616177
# question_well_written : 0.5172281812146284
# answer_helpful : 0.27378370210154007
# answer_level_of_information : 0.45729384382970895
# answer_plausible : 0.1726369612651556
# answer_relevance : 0.20321320547230098
# answer_satisfaction : 0.3609264160475774
# answer_type_instructions : 0.7714005736196025
# answer_type_procedure : 0.3167391088875858
# answer_type_reason_explanation : 0.69948470127083
# answer_well_written : 0.20999747875341068


# question_asker_intent_understanding : 0.3913240782174862
# question_expect_short_answer : 0.32475526913414987
# question_fact_seeking : 0.3817788468272073
# question_interestingness_others : 0.36998401526413244
# question_not_really_a_question : 0.1289263771516838
# question_type_consequence : 0.25940544311123
# question_type_procedure : 0.37509398018350876
# question_type_spelling : 0.14111060200616177
# answer_helpful : 0.27378370210154007
# answer_plausible : 0.1726369612651556
# answer_relevance : 0.20321320547230098
# answer_satisfaction : 0.3609264160475774
# answer_type_procedure : 0.3167391088875858
# answer_well_written : 0.20999747875341068