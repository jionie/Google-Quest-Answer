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
parser.add_argument('--model_name', type=str, default="bert-base-uncased", \
    required=False, help='specify the model_name for BertTokenizer and Net')
parser.add_argument('--hidden_layers', type=list, default=[-1, -3, -5, -7, -9], \
    required=False, help='specify the hidden_layers for Loss')
parser.add_argument('--optimizer', type=str, default='BertAdam', required=False, help='specify the optimizer')
parser.add_argument("--lr_scheduler", type=str, default='WarmupLinearSchedule', required=False, help="specify the lr scheduler")
parser.add_argument('--loss', type=str, default="bce", required=True, help="specify the loss for training")
parser.add_argument("--batch_size", type=int, default=8, required=False, help="specify the batch size for training")
parser.add_argument("--valid_batch_size", type=int, default=32, required=False, help="specify the batch size for validating")
parser.add_argument('--num_workers', type=int, default=2, \
    required=False, help='specify the num_workers for testing dataloader')
parser.add_argument("--checkpoint_folder", type=str, default="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/model", \
    required=False, help="specify the folder for checkpoint")
parser.add_argument('--seed', type=int, default=42, required=True, help="specify the seed for training")
parser.add_argument('--n_splits', type=int, default=5, required=True, help="specify the n_splits for training")
parser.add_argument('--augment', action='store_true', help="specify whether augmentation for training")



############################################################################## Define Constant
NUM_CLASS = 30
TARGET_COLUMNS = ['question_asker_intent_understanding',
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
                'answer_helpful',
                'answer_level_of_information',
                'answer_plausible',
                'answer_relevance',
                'answer_satisfaction',
                'answer_type_instructions',
                'answer_type_procedure',
                'answer_type_reason_explanation',
                'answer_well_written']


############################################################################## seed All
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHseed'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark     = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.deterministic = True


############################################################################## define function for training
def get_oof(
            n_splits,
            fold,
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
        for key in keys:
            if any(s in key for s in skip): continue
            try:
                state_dict[key] = pretrain_state_dict[key]
            except:
                print(key)
        model.load_state_dict(state_dict)
        
        return model


    ############################################################################### model
    if model_type == "bert":
        model = QuestNet(model_type=model_name, n_classes=NUM_CLASS, hidden_layers=hidden_layers)
    elif model_type == "xlnet":
        model = QuestNet(model_type=model_name, n_classes=NUM_CLASS, hidden_layers=hidden_layers)
    else:
        raise NotImplementedError
    
    model = model.cuda()
    load(model, checkpoint_filepath)

    
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
                       checkpoint_folder
                       ):
    
    for fold in range(n_splits):
        
        val_data_path = train_data_folder + "split/val_fold_%s_seed_%s.csv"%(fold, seed)
        val_df = pd.read_csv(val_data_path)
        pred_val = np.load(checkpoint_folder + 'probability_pred_fold_' + str(fold) + '.uint8.npz')['arr_0']
        
        val_df[TARGET_COLUMNS] = pred_val
        
        if fold == 0:
            oof = val_df.copy()
        else:
            oof = pd.concat([oof, val_df], axis=0)

    for column in ["Unnamed: 0", "Unnamed: 0.1"]:
        if column in oof.columns:
            oof = oof.drop([column], axis=1)
            
    oof = oof.sort_values(by="qa_id")
    oof.to_csv(checkpoint_folder + '/oof.csv')
    
    return


def get_spearman(train_df, oof_df, checkpoint_folder):
    
    oof_df = postprocessing(oof_df)

    spearman = Spearman(train_df[TARGET_COLUMNS].values, oof_df[TARGET_COLUMNS].values)
    
    log = Logger()
    log.open(os.path.join(checkpoint_folder, 'in_fold_validartion.txt'), mode='a+')
    log.write('oof_spearman: %f\n' % \
                (spearman))
    log.write('\n')
    
    return

def postprocessing(oof_df):
    oof_values = oof_df[TARGET_COLUMNS].values
    oof_values = np.round(oof_values * 90) / 90
    oof_df[TARGET_COLUMNS] = oof_values
    return oof_df

if __name__ == "__main__":

    # torch.multiprocessing.set_start_method('spawn')

    args = parser.parse_args()

    seed_everything(args.seed)
    
    
    if args.augment:
        checkpoint_folder = os.path.join(args.checkpoint_folder, args.model_type + '/' +args. model_name + '-' + args.loss + '-' + \
        args.optimizer + '-' + args.lr_scheduler + '-' + str(args.n_splits) + '-' + str(args.seed) + '-' + 'aug_differential_relu/')
    else:
        checkpoint_folder = os.path.join(args.checkpoint_folder, args.model_type + '/' +args. model_name + '-' + args.loss + '-' + \
        args.optimizer + '-' + args.lr_scheduler + '-' + str(args.n_splits) + '-' + str(args.seed) + '/')
    
    # get oof
    
    for fold in range(args.n_splits):
        
        # get train_data_loader and val_data_loader
        train_data_path = args.train_data_folder + "split/train_fold_%s_seed_%s.csv"%(fold, args.seed)
        val_data_path   = args.train_data_folder + "split/val_fold_%s_seed_%s.csv"%(fold, args.seed)

        if ((args.model_type == "bert") or (args.model_type == "xlnet")):
            _, val_data_loader = get_train_val_loaders(train_data_path=train_data_path, \
                                                        val_data_path=val_data_path, \
                                                        model_type=args.model_name, \
                                                        batch_size=args.batch_size, \
                                                        val_batch_size=args.valid_batch_size, \
                                                        num_workers=args.num_workers, \
                                                        augment=args.augment)
        else:
            raise NotImplementedError

    
        get_oof(args.n_splits, \
                fold, \
                val_data_loader, \
                args.model_type, \
                args.model_name, \
                args.hidden_layers, \
                args.valid_batch_size, \
                checkpoint_folder, \
                args.seed)
    
    generate_oof_files(args.train_data_folder, \
                       args.n_splits, \
                       args.seed, \
                       checkpoint_folder)
    
    train_df = pd.read_csv(args.train_data_folder + "train.csv")
    oof_df = pd.read_csv(checkpoint_folder + "/oof.csv")
    
    get_spearman(train_df, oof_df, checkpoint_folder)

    gc.collect()
