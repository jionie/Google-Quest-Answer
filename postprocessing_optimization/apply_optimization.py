
from functools import partial
import pandas as pd
import numpy as np
import sys
from metric import *
from file import *
from include import *

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
    
    if (len(thresholds) == 0):
        return oof_df
    
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

def postprocessing(df, best_thresholds_list, target_columns):
    
    for i in range(len(target_columns)):
        df = apply_threshold(df, target_columns[i], best_thresholds_list[i])
        
    return df
    
    
    
if __name__ == "__main__":
    
    model_folder = "/media/jionie/my_disk/Kaggle/Google_Quest_Answer/model/"
    oof_question_answer = pd.read_csv(model_folder + "bert/bert-base-uncased-Answer-bce-BertAdam-WarmupLinearSchedule-5-2020-aug_differential/oof_question_answer.csv")
    oof_xlnet_base_cased = pd.read_csv(model_folder + "xlnet/xlnet-base-cased-bce-BertAdam-WarmupLinearSchedule-5-1997-aug_differential_relu_v1/oof_xlnet_base_cased.csv")
    oof_bert_base_uncased = pd.read_csv(model_folder + "bert/bert-base-uncased-bce-BertAdam-WarmupLinearSchedule-10-2020-aug_differential_relu_v2/oof_bert_base_uncased.csv")
    oof_bert_base_cased = pd.read_csv(model_folder + "bert/bert-base-cased-bce-BertAdam-WarmupLinearSchedule-10-1996-aug_differential_relu_v2/oof-bert-base-cased-v2.csv")

    # oof_df = (oof_xlnet_base_cased[TARGET_COLUMNS] + oof_bert_base_uncased[TARGET_COLUMNS] \
    #     + oof_bert_base_cased[TARGET_COLUMNS] + oof_question_answer[TARGET_COLUMNS])/4.0
    oof_df = (oof_xlnet_base_cased[TARGET_COLUMNS] + oof_bert_base_uncased[TARGET_COLUMNS] \
        + oof_bert_base_cased[TARGET_COLUMNS])/3.0

    train_df = pd.read_csv("/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/train.csv")
    
    spearman = Spearman(train_df[TARGET_COLUMNS].values, oof_df[TARGET_COLUMNS].values)
    print(spearman)
    
    best_threshold_list = []
    
    # open file and read the content in a list
    with open('best_thresholds.txt', 'r') as filehandle:
        filecontents = filehandle.readlines()

        for line in filecontents:
            # remove linebreak which is the last character of the string
            current_place = line[:-1]
            if len(current_place) == 2:
                best_threshold_list.append([])
            else:
                current_place = current_place.strip('][').split(', ')
                current_place = [float(element) for element in current_place]
                # add item to the list
                best_threshold_list.append(current_place)
    
    oof_df = postprocessing(oof_df, best_threshold_list, TARGET_COLUMNS)
    spearman = Spearman(train_df[TARGET_COLUMNS].values, oof_df[TARGET_COLUMNS].values)
    
    print(spearman)
    oof_df = oof_df[TARGET_COLUMNS]
    oof_df['qa_id'] = train_df['qa_id']
    oof_df.to_csv("oof_after_postprocessing.csv")        
    # test = pd.read_csv("/Users/atanas.atanasov/Downloads/test.csv")

    # optimization_results = pd.read_csv("mygeneratedcoeffs")

    # for col in TARGET_COLUMNS:
    #     interim_result = optimization_results.loc[optimization_results.col==col, 'coeffs'].tolist()
    #     interim_result2 = ' '.join([str(elem) for elem in interim_result])
    #     interim_result2 = interim_result2.strip("[]").split(",")
    #     coeffs = [int(x) for x in interim_result2]

    #     colidx = TARGET_COLUMNS.index(col)
    #     bins = np.percentile(pred[:, colidx], list(set(coeffs)))
    #     pred[:, colidx] = np.digitize(pred[:, colidx], np.sort(bins)) / len(bins)


    # final_result['qa_id'] = test['qa_id']
    # final_result[TARGET_COLUMNS] = pd.DataFrame(pred)

    # final_result.to_csv("submission.csv", index=False)    