# import os and define graphic card
import os
os.environ["OMP_NUM_THREADS"] = "1"

# import common libraries
import gc
import random
import argparse
import pandas as pd
import numpy as np
from functools import partial
from sklearn.preprocessing import MinMaxScaler

# import pytorch related libraries
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, _LRScheduler
from tensorboardX import SummaryWriter
from pytorch_pretrained_bert.optimization import BertAdam
from transformers import get_linear_schedule_with_warmup

# import apex for mix precision training
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from apex.optimizers import FusedAdam

# import dataset class
from dataset.dataset import *

# import utils
from utils.ranger import *
from utils.lrs_scheduler import * 
from utils.loss_function import *
from utils.metric import *
from utils.file import *

# import model
from model.model_bert import *


############################################################################## Define Argument
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--train_data_folder", type=str, default="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/input/google-quest-challenge/", \
    required=False, help="specify the folder for training data")
parser.add_argument('--model_type', type=str, default="bert", \
    required=False, help='specify the model_type for BertTokenizer and Net')
parser.add_argument('--content', type=str, default="Question", \
    required=False, help='specify the content for token')
parser.add_argument('--model_name', type=str, default="bert-base-uncased", \
    required=False, help='specify the model_name for BertTokenizer and Net')
parser.add_argument('--hidden_layers', type=list, default=[-3, -4, -5, -6, -7], \
    required=False, help='specify the hidden_layers for Loss')
parser.add_argument('--optimizer', type=str, default='BertAdam', required=False, help='specify the optimizer')
parser.add_argument("--lr_scheduler", type=str, default='WarmupLinearSchedule', required=False, help="specify the lr scheduler")
parser.add_argument('--loss', type=str, default="bce", required=True, help="specify the loss for training")
parser.add_argument("--batch_size", type=int, default=8, required=False, help="specify the batch size for training")
parser.add_argument("--valid_batch_size", type=int, default=32, required=False, help="specify the batch size for validating")
parser.add_argument('--num_workers', type=int, default=2, \
    required=False, help='specify the num_workers for oof_dfing dataloader')
parser.add_argument("--checkpoint_folder", type=str, default="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/model", \
    required=False, help="specify the folder for checkpoint")
parser.add_argument('--seed', type=int, default=42, required=True, help="specify the seed for training")
parser.add_argument('--n_splits', type=int, default=5, required=True, help="specify the n_splits for training")
parser.add_argument('--augment', action='store_true', help="specify whether augmentation for training")
parser.add_argument('--merge', action='store_true', help="specify whether to merge oof of question and answer")
parser.add_argument('--extra_token', action='store_true', default=False, help='whether to use extra token for extra tasks')



############################################################################## Define Constant
QUESTION_TARGET_COLUMNS = ['question_asker_intent_understanding',
                'question_body_critical',
                'question_conversational',
                'question_expect_short_answer',
                'question_fact_seeking',
                'question_has_commonly_accepted_answer',
                'question_interestingness_others',
                'question_interestingness_self',
                'question_multi_intent',
                'question_not_really_a_question',
                'question_opinion_seeking',
                'question_type_choice',
                'question_type_compare',
                'question_type_consequence',
                'question_type_definition',
                'question_type_entity',
                'question_type_instructions',
                'question_type_procedure',
                'question_type_reason_explanation',
                'question_type_spelling',
                'question_well_written',
                ]

ANSWER_TARGET_COLUMNS = [
                'answer_helpful',
                'answer_level_of_information',
                'answer_plausible',
                'answer_relevance',
                'answer_satisfaction',
                'answer_type_instructions',
                'answer_type_procedure',
                'answer_type_reason_explanation',
                'answer_well_written']

TARGET_COLUMNS = QUESTION_TARGET_COLUMNS + ANSWER_TARGET_COLUMNS


############################################################################## seed All
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHseed'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark     = False  ##uses the inbuilt cudnn auto-tuner to find the fasoof_df convolution algorithms. -
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.deterministic = True


############################################################################## define function for training
def get_oof(
            n_splits,
            fold,
            content,
            val_data_loader,
            model_type,
            model_name,
            hidden_layers, 
            valid_batch_size,
            checkpoint_folder,
            seed
            ):
    
    torch.cuda.empty_cache()

    checkpoint_filename = 'fold_' + str(fold) + "_checkpoint.pth"
    checkpoint_filepath = os.path.join(checkpoint_folder, checkpoint_filename)


    ############################################################################## define unet model with backbone
    def load(model, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = model.state_dict()
        keys = list(state_dict.keys())
        
        # for model trained with dataparallel
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in pretrain_state_dict.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        # pretrain_state_dict = new_state_dict
                
        for key in keys:
            if any(s in key for s in skip): continue
            try:
                state_dict[key] = pretrain_state_dict[key]
            except:
                print(key)
        model.load_state_dict(state_dict)
        
        return model


    ############################################################################### model
    if content == "Question_Answer":
        NUM_CLASS = 30
    elif content == "Question":
        NUM_CLASS = 21
    elif content == "Answer":
        NUM_CLASS = 9
    else:
        raise NotImplementedError
    
    if model_type == "bert":
        model = QuestNet(model_type=model_name, n_classes=NUM_CLASS, hidden_layers=hidden_layers)
    elif model_type == "xlnet":
        model = QuestNet(model_type=model_name, n_classes=NUM_CLASS, hidden_layers=hidden_layers)
    else:
        raise NotImplementedError
    
    model = model.cuda()
    model = load(model, checkpoint_filepath)

    
    # init statistics

    labels_val   = None
    pred_val     = None
    
    with torch.no_grad():
        
        # init cache
        torch.cuda.empty_cache()

        for val_batch_i, (token_ids, seg_ids, labels) in enumerate(val_data_loader):
            
            # set model to eval mode
            model.eval()

            # set input to cuda mode
            token_ids = token_ids.cuda()
            seg_ids   = seg_ids.cuda()
            labels    = labels.cuda().float()

           
            prediction = model(token_ids, seg_ids)  
            prediction = torch.sigmoid(prediction)

            if val_batch_i == 0:
                labels_val = labels.cpu().detach().numpy()
                pred_val   = prediction.cpu().detach().numpy()
            else:
                labels_val = np.concatenate((labels_val, labels.cpu().detach().numpy()), axis=0)
                pred_val   = np.concatenate((pred_val, prediction.cpu().detach().numpy()), axis=0)

            
        spearman   = Spearman(labels_val, pred_val)

    
    print("------------------------Valadation----------------------")
    print("--------------------------------------------------------")
    print("fold", fold, "in fold validation spearman: ", spearman)
    print("--------------------------------------------------------")
    log = Logger()
    log.open(os.path.join(checkpoint_folder, 'in_fold_validartion.txt'), mode='a+')
    log.write('fold: %f val_spearman: %f\n' % \
                (fold, spearman))
    log.write('\n')
    np.savez_compressed(checkpoint_folder + '/probability_label_fold_' + str(fold) + '.uint8.npz', labels_val)
    np.savez_compressed(checkpoint_folder + '/probability_pred_fold_' + str(fold) + '.uint8.npz', pred_val)
    
    
    
def generate_oof_files(train_data_folder, \
                       n_splits, \
                       seed, \
                       checkpoint_folder, \
                       target_columns, \
                       content
                       ):
    
    for fold in range(n_splits):
        
        val_data_path = train_data_folder + "split/val_fold_%s_seed_%s.csv"%(fold, seed)
        val_df = pd.read_csv(val_data_path)
        pred_val = np.load(checkpoint_folder + 'probability_pred_fold_' + str(fold) + '.uint8.npz')['arr_0']
        
        val_df[target_columns] = pred_val
        
        if fold == 0:
            oof = val_df.copy()
        else:
            oof = pd.concat([oof, val_df], axis=0)

    for column in ["Unnamed: 0", "Unnamed: 0.1"]:
        if column in oof.columns:
            oof = oof.drop([column], axis=1)
    save_columns = ["qa_id"]+target_columns
    oof = oof[save_columns]      
    oof = oof.sort_values(by="qa_id")
    oof.to_csv(checkpoint_folder + '/oof_' + content + '.csv')
    
    return


def get_spearman(train_df, oof_df, checkpoint_folder, target_columns):
    
    oof_df = postprocessing(oof_df, target_columns)

    spearman = Spearman(train_df[target_columns].values, oof_df[target_columns].values)
    
    log = Logger()
    log.open(os.path.join(checkpoint_folder, 'in_fold_validartion.txt'), mode='a+')
    log.write('oof_spearman: %f\n' % \
                (spearman))
    log.write('\n')
    
    return

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

if __name__ == "__main__":

    # torch.multiprocessing.set_start_method('spawn')

    args = parser.parse_args()

    seed_everything(args.seed)
    
    
    if args.augment:
        if args.extra_token:
            checkpoint_folder = os.path.join(args.checkpoint_folder, args.model_type + '/' + args.model_name + '-' + args.content + '-' + args.loss + '-' + \
                args.optimizer + '-' + args.lr_scheduler + '-' + str(args.n_splits) + '-' + str(args.seed) + '-' + 'aug_differential_extra_token/')
        else:
            checkpoint_folder = os.path.join(args.checkpoint_folder, args.model_type + '/' + args.model_name + '-' + args.content + '-' + args.loss + '-' + \
                args.optimizer + '-' + args.lr_scheduler + '-' + str(args.n_splits) + '-' + str(args.seed) + '-' + 'aug_differential/')
    else:
        if args.extra_token:
            checkpoint_folder = os.path.join(args.checkpoint_folder, args.model_type + '/' + args.model_name + '-' + args.content + '-' + args.loss + '-' + \
                args.optimizer + '-' + args.lr_scheduler + '-' + str(args.n_splits) + '-' + str(args.seed) + '-' + 'extra_token/')
        else:
            checkpoint_folder = os.path.join(args.checkpoint_folder, args.model_type + '/' + args.model_name + '-' + args.content + '-' + args.loss + '-' + \
                args.optimizer + '-' + args.lr_scheduler + '-' + str(args.n_splits) + '-' + str(args.seed) + '-' + '/')
    
    if args.content == "Question_Answer":
        target_columns = TARGET_COLUMNS
    elif args.content == "Question":
        target_columns = QUESTION_TARGET_COLUMNS
    elif args.content == "Answer":
        target_columns = ANSWER_TARGET_COLUMNS
    else:
        raise NotImplementedError
    
    # get oof
    
    # for fold in range(args.n_splits):
        
    #     # get train_data_loader and val_data_loader
    #     train_data_path = args.train_data_folder + "split/train_fold_%s_seed_%s.csv"%(fold, args.seed)
    #     val_data_path   = args.train_data_folder + "split/val_fold_%s_seed_%s.csv"%(fold, args.seed)

    #     if ((args.model_type == "bert") or (args.model_type == "xlnet")):
    #         _, val_data_loader = get_train_val_loaders(train_data_path=train_data_path, \
    #                                                     val_data_path=val_data_path, \
    #                                                     model_type=args.model_name, \
    #                                                       content=args.content, \
    #                                                     batch_size=args.batch_size, \
    #                                                     val_batch_size=args.valid_batch_size, \
    #                                                     num_workers=args.num_workers, \
    #                                                     augment=args.augment, \
    #                                                     extra_token=False)
    #     else:
    #         raise NotImplementedError

    
    #     get_oof(args.n_splits, \
    #             fold, \
    #             val_data_loader, \
    #             args.model_type, \
    #             args.model_name, \
    #             args.hidden_layers, \
    #             args.valid_batch_size, \
    #             checkpoint_folder, \
    #             args.seed)
    
    generate_oof_files(args.train_data_folder, \
                       args.n_splits, \
                       args.seed, \
                       checkpoint_folder, \
                       target_columns, \
                       args.content)
    
    train_df = pd.read_csv(args.train_data_folder + "train.csv")
    oof_df = pd.read_csv(checkpoint_folder + "/oof.csv")
    
    get_spearman(train_df, oof_df, checkpoint_folder, target_columns)
    
    if args.merge:
        if args.augment:
            if args.extra_token:
                checkpoint_folder_question = os.path.join(args.checkpoint_folder, args.model_type + '/' + args.model_name + '-Question-' + args.loss + '-' + \
                    args.optimizer + '-' + args.lr_scheduler + '-' + str(args.n_splits) + '-' + str(args.seed) + '-' + 'aug_differential_extra_token/')
            else:
                checkpoint_folder_question = os.path.join(args.checkpoint_folder, args.model_type + '/' + args.model_name + '-Question-' + args.loss + '-' + \
                    args.optimizer + '-' + args.lr_scheduler + '-' + str(args.n_splits) + '-' + str(args.seed) + '-' + 'aug_differential/')
        else:
            if args.extra_token:
                checkpoint_folder_question = os.path.join(args.checkpoint_folder, args.model_type + '/' + args.model_name + '-Question-' + args.loss + '-' + \
                    args.optimizer + '-' + args.lr_scheduler + '-' + str(args.n_splits) + '-' + str(args.seed) + '-' + 'extra_token/')
            else:
                checkpoint_folder_question = os.path.join(args.checkpoint_folder, args.model_type + '/' + args.model_name + '-Question-' + args.loss + '-' + \
                    args.optimizer + '-' + args.lr_scheduler + '-' + str(args.n_splits) + '-' + str(args.seed) + '-' + '/')
                
        oof_question = pd.read_csv(checkpoint_folder_question + "oof_Question.csv")
        
        if args.augment:
            if args.extra_token:
                checkpoint_folder_answer = os.path.join(args.checkpoint_folder, args.model_type + '/' + args.model_name + '-Answer-' + args.loss + '-' + \
                    args.optimizer + '-' + args.lr_scheduler + '-' + str(args.n_splits) + '-' + str(args.seed) + '-' + 'aug_differential_extra_token/')
            else:
                checkpoint_folder_answer = os.path.join(args.checkpoint_folder, args.model_type + '/' + args.model_name + '-Answer-' + args.loss + '-' + \
                    args.optimizer + '-' + args.lr_scheduler + '-' + str(args.n_splits) + '-' + str(args.seed) + '-' + 'aug_differential/')
        else:
            if args.extra_token:
                checkpoint_folder_answer = os.path.join(args.checkpoint_folder, args.model_type + '/' + args.model_name + '-Answer-' + args.loss + '-' + \
                    args.optimizer + '-' + args.lr_scheduler + '-' + str(args.n_splits) + '-' + str(args.seed) + '-' + 'extra_token/')
            else:
                checkpoint_folder_answer = os.path.join(args.checkpoint_folder, args.model_type + '/' + args.model_name + '-Answer-' + args.loss + '-' + \
                    args.optimizer + '-' + args.lr_scheduler + '-' + str(args.n_splits) + '-' + str(args.seed) + '-' + '/')
                
        oof_answer = pd.read_csv(checkpoint_folder_answer + "oof_Answer.csv")
        
        oof_df = pd.concat([oof_question[QUESTION_TARGET_COLUMNS], oof_answer[ANSWER_TARGET_COLUMNS]], axis=1)
        oof_df['qa_id'] = oof_question['qa_id']
        
        get_spearman(train_df, oof_df, checkpoint_folder, TARGET_COLUMNS)
        
        oof_df.to_csv(checkpoint_folder_answer + "oof_Question_Answer.csv")

    gc.collect()
