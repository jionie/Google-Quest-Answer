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
parser.add_argument('--content', type=str, default="Question", \
    required=False, help='specify the content for token')
parser.add_argument('--hidden_layers', type=list, default=[-3, -4, -5, -6, -7], \
    required=False, help='specify the hidden_layers for Loss')
parser.add_argument('--optimizer', type=str, default='BertAdam', required=False, help='specify the optimizer')
parser.add_argument("--lr_scheduler", type=str, default='WarmupLinearSchedule', required=False, help="specify the lr scheduler")
parser.add_argument("--warmup_proportion",  type=float, default=0.001, required=False, \
    help="Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10%% of training.")
parser.add_argument("--lr", type=float, default=3e-5, required=False, help="specify the initial learning rate for training")
parser.add_argument("--batch_size", type=int, default=8, required=False, help="specify the batch size for training")
parser.add_argument("--valid_batch_size", type=int, default=32, required=False, help="specify the batch size for validating")
parser.add_argument("--num_epoch", type=int, default=12, required=False, help="specify the total epoch")
parser.add_argument("--accumulation_steps", type=int, default=4, required=False, help="specify the accumulation steps")
parser.add_argument('--num_workers', type=int, default=2, \
    required=False, help='specify the num_workers for testing dataloader')
parser.add_argument("--start_epoch", type=int, default=0, required=False, help="specify the start epoch for continue training")
parser.add_argument("--checkpoint_folder", type=str, default="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/model", \
    required=False, help="specify the folder for checkpoint")
parser.add_argument('--extra_token', action='store_true', default=False, help='whether to use extra token for extra tasks')
parser.add_argument('--load_pretrain', action='store_true', default=False, help='whether to load pretrain model')
parser.add_argument('--fold', type=int, default=0, required=True, help="specify the fold for training")
parser.add_argument('--seed', type=int, default=42, required=True, help="specify the seed for training")
parser.add_argument('--n_splits', type=int, default=5, required=True, help="specify the n_splits for training")
parser.add_argument('--loss', type=str, default="mse", required=True, help="specify the loss for training")
parser.add_argument('--augment', action='store_true', help="specify whether augmentation for training")


############################################################################## Define Constant
NUM_CATEGORY_CLASS=5
NUM_HOST_CLASS=64
AUXILIARY_WEIGHTs = [1, 0.05, 0.05]
DECAY_FACTOR = 0.95
MIN_LR = 2e-6
# UNBALANCE_WEIGIHT = [2, 1, 2, 2, 2, 2, \
#                   1, 2, 2, 4, 1, 2, \
#                   4, 4, 4, 4, 1, 2, \
#                   1, 4, 2, 2, 2, 2, \
#                   2, 1, 1, 2, 1, 2]

UNBALANCE_WEIGIHT = [1, 1, 1, 1, 1, 1, \
                     1, 1, 1, 2, 1, 1, \
                     2, 2, 2, 2, 1, 1, \
                     1, 1, 1, 1, 1, 1, \
                     1, 1, 1, 1, 1, 1]

TRAING_WEIGIHT = [i for i in UNBALANCE_WEIGIHT]
# 1 is balanced, 2 is unbalanced, 3 is extremely unbalanced


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
def training(
            content,
            n_splits,
            fold,
            train_data_loader, 
            val_data_loader,
            model_type,
            model_name,
            hidden_layers, 
            optimizer_name,
            lr_scheduler_name,
            lr,
            warmup_proportion,
            batch_size,
            valid_batch_size,
            num_epoch,
            start_epoch,
            accumulation_steps,
            checkpoint_folder,
            load_pretrain,
            seed,
            loss,
            extra_token, 
            augment
            ):
    
    torch.cuda.empty_cache()

    COMMON_STRING ='@%s:  \n' % os.path.basename(__file__)
    COMMON_STRING += '\tset random seed\n'
    COMMON_STRING += '\t\tseed = %d\n'%seed

    COMMON_STRING += '\tset cuda environment\n'
    COMMON_STRING += '\t\ttorch.__version__              = %s\n'%torch.__version__
    COMMON_STRING += '\t\ttorch.version.cuda             = %s\n'%torch.version.cuda
    COMMON_STRING += '\t\ttorch.backends.cudnn.version() = %s\n'%torch.backends.cudnn.version()
    try:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = %s\n'%os.environ['CUDA_VISIBLE_DEVICES']
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = None\n'
        NUM_CUDA_DEVICES = 1

    COMMON_STRING += '\t\ttorch.cuda.device_count()      = %d\n'%torch.cuda.device_count()
    COMMON_STRING += '\n'
    
    if augment:
        if extra_token:
            checkpoint_folder = os.path.join(checkpoint_folder, model_type + '/' + model_name + '-' + loss + '-' + \
                optimizer_name + '-' + lr_scheduler_name + '-' + str(n_splits) + '-' + str(seed) + '-' + 'aug_differential_extra_token/')
        else:
            checkpoint_folder = os.path.join(checkpoint_folder, model_type + '/' + model_name + '-' + loss + '-' + \
                optimizer_name + '-' + lr_scheduler_name + '-' + str(n_splits) + '-' + str(seed) + '-' + 'aug_differential/')
    else:
        if extra_token:
            checkpoint_folder = os.path.join(checkpoint_folder, model_type + '/' + model_name + '-' + loss + '-' + \
                optimizer_name + '-' + lr_scheduler_name + '-' + str(n_splits) + '-' + str(seed) + '-' + 'extra_token/')
        else:
            checkpoint_folder = os.path.join(checkpoint_folder, model_type + '/' + model_name + '-' + loss + '-' + \
                optimizer_name + '-' + lr_scheduler_name + '-' + str(n_splits) + '-' + str(seed) + '-' + '/')

    checkpoint_filename = 'fold_' + str(fold) + "_checkpoint.pth"
    checkpoint_filepath = os.path.join(checkpoint_folder, checkpoint_filename)

    os.makedirs(checkpoint_folder, exist_ok=True)
    
    log = Logger()
    log.open(os.path.join(checkpoint_folder, 'fold_' + str(fold) + '_log_train.txt'), mode='a+')
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tseed         = %u\n' % seed)
    log.write('\tFOLD         = %s\n' % fold)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % checkpoint_folder)
    log.write('\n')


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
    if content == "Question_Answer":
        NUM_CLASS = 30
    elif content == "Question":
        NUM_CLASS = 21
    elif content == "Answer":
        NUM_CLASS = 9
    else:
        raise NotImplementedError
        
        
    if model_type == "bert":
        if extra_token:
            model = QuestNet(model_type=model_name, \
                    n_classes=NUM_CLASS, \
                    n_category_classes=NUM_CATEGORY_CLASS, \
                    n_host_classes=NUM_HOST_CLASS, \
                    hidden_layers=hidden_layers, \
                    extra_token=True)
        else:
            model = QuestNet(model_type=model_name, \
                    n_classes=NUM_CLASS, \
                    n_category_classes=NUM_CATEGORY_CLASS, \
                    n_host_classes=NUM_HOST_CLASS, \
                    hidden_layers=hidden_layers, \
                    extra_token=False)
    elif model_type == "xlnet":
        if extra_token:
            model = QuestNet(model_type=model_name, \
                    n_classes=NUM_CLASS, \
                    n_category_classes=NUM_CATEGORY_CLASS, \
                    n_host_classes=NUM_HOST_CLASS, \
                    hidden_layers=hidden_layers, \
                    extra_token=True)
        else:
            model = QuestNet(model_type=model_name, \
                    n_classes=NUM_CLASS, \
                    n_category_classes=NUM_CATEGORY_CLASS, \
                    n_host_classes=NUM_HOST_CLASS, \
                    hidden_layers=hidden_layers, \
                    extra_token=False)
    else:
        raise NotImplementedError
    
    model = model.cuda()
    
    if load_pretrain:
        load(model, checkpoint_filepath)

    ############################################################################### optimizer
    if ((model_type == "bert") or (model_type == "xlnet")) :
        
        optimizer_grouped_parameters = []
        list_lr = []
        
        if ((model_name == "bert-base-uncased") or (model_name == "bert-base-cased")):

            list_layers = [model.bert_model.embeddings,
                      model.bert_model.encoder.layer[0],
                      model.bert_model.encoder.layer[1],
                      model.bert_model.encoder.layer[2],
                      model.bert_model.encoder.layer[3],
                      model.bert_model.encoder.layer[4],
                      model.bert_model.encoder.layer[5],
                      model.bert_model.encoder.layer[6],
                      model.bert_model.encoder.layer[7],
                      model.bert_model.encoder.layer[8],
                      model.bert_model.encoder.layer[9],
                      model.bert_model.encoder.layer[10],
                      model.bert_model.encoder.layer[11],
                      model.fc_1,
                      model.fc
                      ]
            
        elif ((model_name == "bert-large-uncased") or (model_name == "bert-large-cased")):
        
            list_layers = [model.bert_model.embeddings,
                  model.bert_model.encoder.layer[0],
                  model.bert_model.encoder.layer[1],
                  model.bert_model.encoder.layer[2],
                  model.bert_model.encoder.layer[3],
                  model.bert_model.encoder.layer[4],
                  model.bert_model.encoder.layer[5],
                  model.bert_model.encoder.layer[6],
                  model.bert_model.encoder.layer[7],
                  model.bert_model.encoder.layer[8],
                  model.bert_model.encoder.layer[9],
                  model.bert_model.encoder.layer[10],
                  model.bert_model.encoder.layer[11],
                  model.bert_model.encoder.layer[12],
                  model.bert_model.encoder.layer[13],
                  model.bert_model.encoder.layer[14],
                  model.bert_model.encoder.layer[15],
                  model.bert_model.encoder.layer[16],
                  model.bert_model.encoder.layer[17],
                  model.bert_model.encoder.layer[18],
                  model.bert_model.encoder.layer[19],
                  model.bert_model.encoder.layer[20],
                  model.bert_model.encoder.layer[21],
                  model.bert_model.encoder.layer[22],
                  model.bert_model.encoder.layer[23],
                  model.fc_1,
                  model.fc
                  ]
            
        elif ((model_name == "flaubert-base-uncased") or (model_name == "flaubert-base-cased")):
    
            list_layers = [
                      model.flaubert_model.position_embeddings,
                      model.flaubert_model.embeddings,
                      model.flaubert_model.layer_norm_emb,
                      [model.flaubert_model.attentions[0], model.flaubert_model.layer_norm1[0],  model.flaubert_model.ffns[0], model.flaubert_model.layer_norm2[0]],
                      [model.flaubert_model.attentions[1], model.flaubert_model.layer_norm1[1],  model.flaubert_model.ffns[1], model.flaubert_model.layer_norm2[1]],
                      [model.flaubert_model.attentions[2], model.flaubert_model.layer_norm1[2],  model.flaubert_model.ffns[2], model.flaubert_model.layer_norm2[2]],
                      [model.flaubert_model.attentions[3], model.flaubert_model.layer_norm1[3],  model.flaubert_model.ffns[3], model.flaubert_model.layer_norm2[3]],
                      [model.flaubert_model.attentions[4], model.flaubert_model.layer_norm1[4],  model.flaubert_model.ffns[4], model.flaubert_model.layer_norm2[4]],
                      [model.flaubert_model.attentions[5], model.flaubert_model.layer_norm1[5],  model.flaubert_model.ffns[5], model.flaubert_model.layer_norm2[5]],
                      [model.flaubert_model.attentions[6], model.flaubert_model.layer_norm1[6],  model.flaubert_model.ffns[6], model.flaubert_model.layer_norm2[6]],
                      [model.flaubert_model.attentions[7], model.flaubert_model.layer_norm1[7],  model.flaubert_model.ffns[7], model.flaubert_model.layer_norm2[7]],
                      [model.flaubert_model.attentions[8], model.flaubert_model.layer_norm1[8],  model.flaubert_model.ffns[8], model.flaubert_model.layer_norm2[8]],
                      [model.flaubert_model.attentions[9], model.flaubert_model.layer_norm1[9],  model.flaubert_model.ffns[9], model.flaubert_model.layer_norm2[9]],
                      [model.flaubert_model.attentions[10], model.flaubert_model.layer_norm1[10],  model.flaubert_model.ffns[10], model.flaubert_model.layer_norm2[10]],
                      [model.flaubert_model.attentions[11], model.flaubert_model.layer_norm1[11],  model.flaubert_model.ffns[11], model.flaubert_model.layer_norm2[11]],
                      model.fc_1,
                      model.fc
                      ]
            
        elif ((model_name == "flaubert-large-cased")):
        
            list_layers = [
                      model.flaubert_model.position_embeddings,
                      model.flaubert_model.embeddings,
                      model.flaubert_model.layer_norm_emb,
                      [model.flaubert_model.attentions[0], model.flaubert_model.layer_norm1[0],  model.flaubert_model.ffns[0], model.flaubert_model.layer_norm2[0]],
                      [model.flaubert_model.attentions[1], model.flaubert_model.layer_norm1[1],  model.flaubert_model.ffns[1], model.flaubert_model.layer_norm2[1]],
                      [model.flaubert_model.attentions[2], model.flaubert_model.layer_norm1[2],  model.flaubert_model.ffns[2], model.flaubert_model.layer_norm2[2]],
                      [model.flaubert_model.attentions[3], model.flaubert_model.layer_norm1[3],  model.flaubert_model.ffns[3], model.flaubert_model.layer_norm2[3]],
                      [model.flaubert_model.attentions[4], model.flaubert_model.layer_norm1[4],  model.flaubert_model.ffns[4], model.flaubert_model.layer_norm2[4]],
                      [model.flaubert_model.attentions[5], model.flaubert_model.layer_norm1[5],  model.flaubert_model.ffns[5], model.flaubert_model.layer_norm2[5]],
                      [model.flaubert_model.attentions[6], model.flaubert_model.layer_norm1[6],  model.flaubert_model.ffns[6], model.flaubert_model.layer_norm2[6]],
                      [model.flaubert_model.attentions[7], model.flaubert_model.layer_norm1[7],  model.flaubert_model.ffns[7], model.flaubert_model.layer_norm2[7]],
                      [model.flaubert_model.attentions[8], model.flaubert_model.layer_norm1[8],  model.flaubert_model.ffns[8], model.flaubert_model.layer_norm2[8]],
                      [model.flaubert_model.attentions[9], model.flaubert_model.layer_norm1[9],  model.flaubert_model.ffns[9], model.flaubert_model.layer_norm2[9]],
                      [model.flaubert_model.attentions[10], model.flaubert_model.layer_norm1[10],  model.flaubert_model.ffns[10], model.flaubert_model.layer_norm2[10]],
                      [model.flaubert_model.attentions[11], model.flaubert_model.layer_norm1[11],  model.flaubert_model.ffns[11], model.flaubert_model.layer_norm2[11]],
                      [model.flaubert_model.attentions[12], model.flaubert_model.layer_norm1[12],  model.flaubert_model.ffns[12], model.flaubert_model.layer_norm2[12]],
                      [model.flaubert_model.attentions[13], model.flaubert_model.layer_norm1[13],  model.flaubert_model.ffns[13], model.flaubert_model.layer_norm2[13]],
                      [model.flaubert_model.attentions[14], model.flaubert_model.layer_norm1[14],  model.flaubert_model.ffns[14], model.flaubert_model.layer_norm2[14]],
                      [model.flaubert_model.attentions[15], model.flaubert_model.layer_norm1[15],  model.flaubert_model.ffns[15], model.flaubert_model.layer_norm2[15]],
                      [model.flaubert_model.attentions[16], model.flaubert_model.layer_norm1[16],  model.flaubert_model.ffns[16], model.flaubert_model.layer_norm2[16]],
                      [model.flaubert_model.attentions[17], model.flaubert_model.layer_norm1[17],  model.flaubert_model.ffns[17], model.flaubert_model.layer_norm2[17]],
                      [model.flaubert_model.attentions[18], model.flaubert_model.layer_norm1[18],  model.flaubert_model.ffns[18], model.flaubert_model.layer_norm2[18]],
                      [model.flaubert_model.attentions[19], model.flaubert_model.layer_norm1[19],  model.flaubert_model.ffns[19], model.flaubert_model.layer_norm2[19]],
                      [model.flaubert_model.attentions[20], model.flaubert_model.layer_norm1[20],  model.flaubert_model.ffns[20], model.flaubert_model.layer_norm2[20]],
                      [model.flaubert_model.attentions[21], model.flaubert_model.layer_norm1[21],  model.flaubert_model.ffns[21], model.flaubert_model.layer_norm2[21]],
                      [model.flaubert_model.attentions[22], model.flaubert_model.layer_norm1[22],  model.flaubert_model.ffns[22], model.flaubert_model.layer_norm2[22]],
                      [model.flaubert_model.attentions[23], model.flaubert_model.layer_norm1[23],  model.flaubert_model.ffns[23], model.flaubert_model.layer_norm2[23]],
                      model.fc_1,
                      model.fc
                      ]
            
        elif (model_name == "xlnet-base-cased"):
            
            list_layers = [model.xlnet_model.word_embedding,
                      model.xlnet_model.layer[0],
                      model.xlnet_model.layer[1],
                      model.xlnet_model.layer[2],
                      model.xlnet_model.layer[3],
                      model.xlnet_model.layer[4],
                      model.xlnet_model.layer[5],
                      model.xlnet_model.layer[6],
                      model.xlnet_model.layer[7],
                      model.xlnet_model.layer[8],
                      model.xlnet_model.layer[9],
                      model.xlnet_model.layer[10],
                      model.xlnet_model.layer[11],
                      model.fc_1,
                      model.fc
                      ]
            
        elif (model_name == "xlnet-large-cased"):
            
            list_layers = [model.xlnet_model.word_embedding,
                      model.xlnet_model.layer[0],
                      model.xlnet_model.layer[1],
                      model.xlnet_model.layer[2],
                      model.xlnet_model.layer[3],
                      model.xlnet_model.layer[4],
                      model.xlnet_model.layer[5],
                      model.xlnet_model.layer[6],
                      model.xlnet_model.layer[7],
                      model.xlnet_model.layer[8],
                      model.xlnet_model.layer[9],
                      model.xlnet_model.layer[10],
                      model.xlnet_model.layer[11],
                      model.xlnet_model.layer[12],
                      model.xlnet_model.layer[13],
                      model.xlnet_model.layer[14],
                      model.xlnet_model.layer[15],
                      model.xlnet_model.layer[16],
                      model.xlnet_model.layer[17],
                      model.xlnet_model.layer[18],
                      model.xlnet_model.layer[19],
                      model.xlnet_model.layer[20],
                      model.xlnet_model.layer[21],
                      model.xlnet_model.layer[22],
                      model.xlnet_model.layer[23],
                      model.fc_1,
                      model.fc
                      ]
            
        elif (model_name == "roberta-base"):
            
            list_layers = [model.roberta_model.word_embedding,
                      model.roberta_model.layer[0],
                      model.roberta_model.layer[1],
                      model.roberta_model.layer[2],
                      model.roberta_model.layer[3],
                      model.roberta_model.layer[4],
                      model.roberta_model.layer[5],
                      model.roberta_model.layer[6],
                      model.roberta_model.layer[7],
                      model.roberta_model.layer[8],
                      model.roberta_model.layer[9],
                      model.roberta_model.layer[10],
                      model.roberta_model.layer[11],
                      model.fc_1,
                      model.fc
                      ]
            
        elif ((model_name == "albert-base-v2") or \
            (model_name == "albert-large-v2") or \
            (model_name == "albert-xlarge-v2") or \
            (model_name == "albert-xxlarge-v2")):
            list_layers = [model.albert_model.embeddings,
                        #    model.albert_model.encoder.embedding_hidden_mapping_in,
                        #    model.albert_model.encoder.albert_layer_groups,
                           model.albert_model.encoder,
                           model.fc_1,
                           model.fc
                           ]
            print("differential lr for ", model_name)

        elif (model_name == "gpt2"):
            list_layers = [# model.gpt2_model.wte,
                           # model.gpt2_model.wpe,
                           model.gpt2_model.h[0],
                           model.gpt2_model.h[1],
                           model.gpt2_model.h[2],
                           model.gpt2_model.h[3],
                           model.gpt2_model.h[4],
                           model.gpt2_model.h[5],
                           model.gpt2_model.h[6],
                           model.gpt2_model.h[7],
                           model.gpt2_model.h[8],
                           model.gpt2_model.h[9],
                           model.gpt2_model.h[10],
                           model.gpt2_model.h[11],
                           model.fc_1,
                           model.fc
                           ]

        else:
            raise NotImplementedError

#         for i in range(len(list_layers)):
#             list_lr.append(lr)
#             lr = lr * DECAY_FACTOR
#             list_lr.append(lr - i * (lr - MIN_LR) / (len(list_layers) - 1))

#         list_lr.reverse()
        
        mult = lr / MIN_LR
        step = mult**(1/(len(list_layers)-1))
        list_lr = [MIN_LR * (step ** i) for i in range(len(list_layers))]
        
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            
        print(list_lr)
        for i in range(len(list_lr)):
            
            if isinstance(list_layers[i], list):
                
                for list_layer in list_layers[i]:
                    layer_parameters = list(list_layer.named_parameters())

                    optimizer_grouped_parameters.append({ \
                        'params': [p for n, p in layer_parameters if not any(nd in n for nd in no_decay)], \
                        'lr': list_lr[i], \
                        'weight_decay': 0.01})

                    optimizer_grouped_parameters.append({ \
                        'params': [p for n, p in layer_parameters if any(nd in n for nd in no_decay)], \
                        'lr': list_lr[i], \
                        'weight_decay': 0.0}) 
                
            else:
            
                layer_parameters = list(list_layers[i].named_parameters())

                optimizer_grouped_parameters.append({ \
                    'params': [p for n, p in layer_parameters if not any(nd in n for nd in no_decay)], \
                    'lr': list_lr[i], \
                    'weight_decay': 0.01})

                optimizer_grouped_parameters.append({ \
                    'params': [p for n, p in layer_parameters if any(nd in n for nd in no_decay)], \
                    'lr': list_lr[i], \
                    'weight_decay': 0.0}) 
            
        if extra_token:
            # add extra fcs
            layer_parameters = list(model.fc_1_category.named_parameters())
            optimizer_grouped_parameters.append({ \
                'params': [p for n, p in layer_parameters if not any(nd in n for nd in no_decay)], \
                'lr': 1e-6, \
                'weight_decay': 0.01})
            optimizer_grouped_parameters.append({ \
                'params': [p for n, p in layer_parameters if any(nd in n for nd in no_decay)], \
                'lr': 1e-6, \
                'weight_decay': 0.0}) 
            layer_parameters = list(model.fc_1_host.named_parameters())
            optimizer_grouped_parameters.append({ \
                'params': [p for n, p in layer_parameters if not any(nd in n for nd in no_decay)], \
                'lr': 1e-6, \
                'weight_decay': 0.01})
            optimizer_grouped_parameters.append({ \
                'params': [p for n, p in layer_parameters if any(nd in n for nd in no_decay)], \
                'lr': 1e-6, \
                'weight_decay': 0.0})
            
            layer_parameters = list(model.fc_category.named_parameters())
            optimizer_grouped_parameters.append({ \
                'params': [p for n, p in layer_parameters if not any(nd in n for nd in no_decay)], \
                'lr': 1e-6, \
                'weight_decay': 0.01})
            optimizer_grouped_parameters.append({ \
                'params': [p for n, p in layer_parameters if any(nd in n for nd in no_decay)], \
                'lr': 1e-6, \
                'weight_decay': 0.0}) 
            layer_parameters = list(model.fc_host.named_parameters())
            optimizer_grouped_parameters.append({ \
                'params': [p for n, p in layer_parameters if not any(nd in n for nd in no_decay)], \
                'lr': 1e-6, \
                'weight_decay': 0.01})
            optimizer_grouped_parameters.append({ \
                'params': [p for n, p in layer_parameters if any(nd in n for nd in no_decay)], \
                'lr': 1e-6, \
                'weight_decay': 0.0})
            
        print("Differential Learning Rate!!")
    
    else:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], \
             'lr': lr, \
             'weight_decay': 0.01}, \
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], \
             'lr': lr, \
             'weight_decay': 0.0}
        ]

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters)
    elif optimizer_name == "Ranger":
        optimizer = Ranger(optimizer_grouped_parameters)
    elif optimizer_name == "BertAdam":
        num_train_optimization_steps = num_epoch * len(train_data_loader) // accumulation_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)
    elif optimizer_name == "FusedAdam":
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              bias_correction=False)
    else:
        raise NotImplementedError
    
    ############################################################################### lr_scheduler
    if lr_scheduler_name == "CosineAnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=1e-5, last_epoch=-1)
        lr_scheduler_each_iter = False
    elif lr_scheduler_name == "WarmRestart":
        scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-6)
        lr_scheduler_each_iter = False
    elif lr_scheduler_name == "WarmupLinearSchedule":
        num_train_optimization_steps = num_epoch * len(train_data_loader) // accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, \
                                        num_warmup_steps=int(num_train_optimization_steps*warmup_proportion), \
                                        num_training_steps=num_train_optimization_steps)
        lr_scheduler_each_iter = True
    else:
        raise NotImplementedError

    log.write('net\n  %s\n'%(model_name))
    log.write('optimizer\n  %s\n'%(optimizer_name))
    log.write('schduler\n  %s\n'%(lr_scheduler_name))
    log.write('\n')

    ###############################################################################  mix precision
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    # model = nn.DataParallel(model)

    ############################################################################### eval setting
    eval_step = len(train_data_loader) # or len(train_data_loader) 
    log_step = 50
    eval_count = 0

    ############################################################################### training
    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  accumulation_steps=%d\n'%(batch_size, accumulation_steps))
    log.write('   experiment  = %s\n' % str(__file__.split('/')[-2:]))
    
    valid_loss = np.zeros(1, np.float32)
    train_loss = np.zeros(1, np.float32)
    valid_metric_optimal = -np.inf
    
    # define tensorboard writer and timer
    writer = SummaryWriter()
    
    # define criterion
    if loss == 'mse':
        criterion = MSELoss()
    elif loss == 'bce':
        # weights = torch.tensor(np.array(TRAING_WEIGIHT) / np.sum(TRAING_WEIGIHT) * 30, dtype=torch.float64).cuda()
        # weights = torch.tensor(np.array(TRAING_WEIGIHT), dtype=torch.float64).cuda()
        # criterion = nn.BCEWithLogitsLoss(weight=weights)
        criterion = nn.BCEWithLogitsLoss()
        criterion_extra = nn.BCEWithLogitsLoss()
    elif loss == 'mse-bce':
        criterion = MSEBCELoss()
    elif loss == 'focal':
        criterion = FocalLoss()
    else:
        raise NotImplementedError
    
    for epoch in range(1, num_epoch+1):

        # init in-epoch statistics
        labels_train = None
        pred_train   = None
        labels_val   = None
        pred_val     = None
        
        # update lr and start from start_epoch  
        if ((epoch > 1) and (not lr_scheduler_each_iter)):
            scheduler.step()
           
        if (epoch < start_epoch):
            continue
        
        log.write("Epoch%s\n" % epoch)
        log.write('\n')

        sum_train_loss = np.zeros_like(train_loss)
        sum_train = np.zeros_like(train_loss)

        # init optimizer
        torch.cuda.empty_cache()
        model.zero_grad()
        
        if extra_token:
            for tr_batch_i, (token_ids, seg_ids, labels, labels_category, labels_host) in enumerate(train_data_loader):
                rate = 0
                for param_group in optimizer.param_groups:
                    rate += param_group['lr'] / len(optimizer.param_groups)
                    
                # set model training mode
                model.train() 

                # set input to cuda mode
                token_ids = token_ids.cuda()
                seg_ids   = seg_ids.cuda()
                labels    = labels.cuda().float()
                labels_category = labels_category.cuda().float()
                labels_host     = labels_host.cuda().float()

                # predict and calculate loss (only need torch.sigmoid when inference)
                prediction, prediction_category, prediction_host = model(token_ids, seg_ids)  
                
                # print(prediction.shape, prediction_category.shape, prediction_host.shape)
                # print(labels.shape, labels_category.shape, labels_host.shape)
                
                loss = AUXILIARY_WEIGHTs[0]*criterion(prediction, labels) + \
                       AUXILIARY_WEIGHTs[1]*criterion_extra(prediction_category, labels_category) + \
                       AUXILIARY_WEIGHTs[2]*criterion_extra(prediction_host, labels_host) 
                
                # use apex
                with amp.scale_loss(loss/accumulation_steps, optimizer) as scaled_loss:
                    scaled_loss.backward()

                # don't use apex
                #loss.backward()
            
                if ((tr_batch_i+1) % accumulation_steps == 0):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
                    optimizer.step()
                    model.zero_grad()
                    # adjust lr
                    if (lr_scheduler_each_iter):
                        scheduler.step()

                    writer.add_scalar('train_loss_' + str(fold), loss.item(), (epoch-1)*len(train_data_loader)*batch_size+tr_batch_i*batch_size)
                
                # calculate statistics
                prediction = torch.sigmoid(prediction)

                if tr_batch_i == 0:
                    labels_train = labels.cpu().detach().numpy()
                    pred_train   = prediction.cpu().detach().numpy()
                else:
                    labels_train = np.concatenate((labels_train, labels.cpu().detach().numpy()), axis=0)
                    pred_train   = np.concatenate((pred_train, prediction.cpu().detach().numpy()), axis=0)
                
                l = np.array([loss.item() * batch_size])
                n = np.array([batch_size])
                sum_train_loss = sum_train_loss + l
                sum_train      = sum_train + n
                
                # log for training
                if (tr_batch_i+1) % log_step == 0:  
                    train_loss          = sum_train_loss / (sum_train + 1e-12)
                    sum_train_loss[...] = 0
                    sum_train[...]      = 0
                    spearman            = Spearman(labels_train, pred_train)
                    log.write('lr: %f train loss: %f train_spearman: %f\n' % \
                        (rate, train_loss[0], spearman))
                
                if (tr_batch_i+1) % eval_step == 0:  
                    
                    eval_count += 1
                    
                    valid_loss = np.zeros(1, np.float32)
                    valid_num  = np.zeros_like(valid_loss)
                    
                    with torch.no_grad():
                        
                        # init cache
                        torch.cuda.empty_cache()

                        for val_batch_i, (token_ids, seg_ids, labels, labels_category, labels_host) in enumerate(val_data_loader):
                            
                            # set model to eval mode
                            model.eval()

                            # set input to cuda mode
                            token_ids = token_ids.cuda()
                            seg_ids   = seg_ids.cuda()
                            labels    = labels.cuda().float()
                            labels_category = labels_category.cuda().float()
                            labels_host     = labels_host.cuda().float()

                            # predict and calculate loss (only need torch.sigmoid when inference)
                            prediction, prediction_category, prediction_host = model(token_ids, seg_ids)  
                            loss = AUXILIARY_WEIGHTs[0]*criterion(prediction, labels) + \
                                    AUXILIARY_WEIGHTs[1]*criterion_extra(prediction_category, labels_category) + \
                                    AUXILIARY_WEIGHTs[2]*criterion_extra(prediction_host, labels_host) 
                                
                            writer.add_scalar('val_loss_' + str(fold), loss.item(), (eval_count-1)*len(val_data_loader)*valid_batch_size+val_batch_i*valid_batch_size)
                            
                            # calculate statistics
                            prediction = torch.sigmoid(prediction)

                            if val_batch_i == 0:
                                labels_val = labels.cpu().detach().numpy()
                                pred_val   = prediction.cpu().detach().numpy()
                            else:
                                labels_val = np.concatenate((labels_val, labels.cpu().detach().numpy()), axis=0)
                                pred_val   = np.concatenate((pred_val, prediction.cpu().detach().numpy()), axis=0)

                            l = np.array([loss.item()*valid_batch_size])
                            n = np.array([valid_batch_size])
                            valid_loss = valid_loss + l
                            valid_num  = valid_num + n
                            
                        valid_loss = valid_loss / valid_num
                        spearman   = Spearman(labels_val, pred_val)

                        log.write('validation loss: %f val_spearman: %f\n' % \
                        (valid_loss[0], spearman))
        else:
            for tr_batch_i, (token_ids, seg_ids, labels) in enumerate(train_data_loader):
                rate = 0
                for param_group in optimizer.param_groups:
                    rate += param_group['lr'] / len(optimizer.param_groups)
                    
                # set model training mode
                model.train() 

                # set input to cuda mode
                token_ids = token_ids.cuda()
                seg_ids   = seg_ids.cuda()
                labels    = labels.cuda().float()

                # predict and calculate loss (only need torch.sigmoid when inference)
                prediction = model(token_ids, seg_ids)  
                loss = criterion(prediction, labels)
                
                # use apex
                with amp.scale_loss(loss/accumulation_steps, optimizer) as scaled_loss:
                    scaled_loss.backward()

                # don't use apex
                #loss.backward()
            
                if ((tr_batch_i+1) % accumulation_steps == 0):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
                    optimizer.step()
                    model.zero_grad()
                    # adjust lr
                    if (lr_scheduler_each_iter):
                        scheduler.step()

                    writer.add_scalar('train_loss_' + str(fold), loss.item(), (epoch-1)*len(train_data_loader)*batch_size+tr_batch_i*batch_size)
                
                # calculate statistics
                prediction = torch.sigmoid(prediction)

                if tr_batch_i == 0:
                    labels_train = labels.cpu().detach().numpy()
                    pred_train   = prediction.cpu().detach().numpy()
                else:
                    labels_train = np.concatenate((labels_train, labels.cpu().detach().numpy()), axis=0)
                    pred_train   = np.concatenate((pred_train, prediction.cpu().detach().numpy()), axis=0)
                
                l = np.array([loss.item() * batch_size])
                n = np.array([batch_size])
                sum_train_loss = sum_train_loss + l
                sum_train      = sum_train + n
                
                # log for training
                if (tr_batch_i+1) % log_step == 0:  
                    train_loss          = sum_train_loss / (sum_train + 1e-12)
                    sum_train_loss[...] = 0
                    sum_train[...]      = 0
                    spearman            = Spearman(labels_train, pred_train)
                    log.write('lr: %f train loss: %f train_spearman: %f\n' % \
                        (rate, train_loss[0], spearman))
                
                if (tr_batch_i+1) % eval_step == 0:  
                    
                    eval_count += 1
                    
                    valid_loss = np.zeros(1, np.float32)
                    valid_num  = np.zeros_like(valid_loss)
                    
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

                            # predict and calculate loss (only need torch.sigmoid when inference)
                            prediction = model(token_ids, seg_ids)  
                            loss = criterion(prediction, labels)
                                
                            writer.add_scalar('val_loss_' + str(fold), loss.item(), (eval_count-1)*len(val_data_loader)*valid_batch_size+val_batch_i*valid_batch_size)
                            
                            # calculate statistics
                            prediction = torch.sigmoid(prediction)

                            if val_batch_i == 0:
                                labels_val = labels.cpu().detach().numpy()
                                pred_val   = prediction.cpu().detach().numpy()
                            else:
                                labels_val = np.concatenate((labels_val, labels.cpu().detach().numpy()), axis=0)
                                pred_val   = np.concatenate((pred_val, prediction.cpu().detach().numpy()), axis=0)

                            l = np.array([loss.item()*valid_batch_size])
                            n = np.array([valid_batch_size])
                            valid_loss = valid_loss + l
                            valid_num  = valid_num + n
                            
                        valid_loss = valid_loss / valid_num
                        spearman   = Spearman(labels_val, pred_val)

                        log.write('validation loss: %f val_spearman: %f\n' % \
                        (valid_loss[0], spearman))

        val_metric_epoch = spearman

        if (val_metric_epoch >= valid_metric_optimal):
            
            log.write('Validation metric improved ({:.6f} --> {:.6f}).  Saving model ...'.format(\
                    valid_metric_optimal, val_metric_epoch))

            valid_metric_optimal = val_metric_epoch
            torch.save(model.state_dict(), checkpoint_filepath)
        
            np.savez_compressed(checkpoint_folder + '/probability_label_fold_' + str(fold) + '.uint8.npz', labels_val)
            np.savez_compressed(checkpoint_folder + '/probability_pred_fold_' + str(fold) + '.uint8.npz', pred_val)
    


if __name__ == "__main__":

    # torch.multiprocessing.set_start_method('spawn')

    args = parser.parse_args()

    seed_everything(args.seed)

    # get train val split
    data_path = args.train_data_folder + "train_augment_final_with_clean.csv"
    get_train_val_split(data_path=data_path, \
                        save_path=args.train_data_folder, \
                        n_splits=args.n_splits, \
                        seed=args.seed)

    # get train_data_loader and val_data_loader
    train_data_path = args.train_data_folder + "split/train_fold_%s_seed_%s.csv"%(args.fold, args.seed)
    val_data_path   = args.train_data_folder + "split/val_fold_%s_seed_%s.csv"%(args.fold, args.seed)

    if ((args.model_type == "bert") or (args.model_type == "xlnet")):
        
        if args.extra_token:
            test_data_path = args.train_data_folder + "test.csv"
            train_df = pd.read_csv(data_path)
            test_df = pd.read_csv(test_data_path)
            
            train_host_list = train_df['host'].unique().tolist()
            test_host_list = test_df['host'].unique().tolist()
            host_encoder = LabelBinarizer()
            host_encoder.fit(list(set(train_host_list + test_host_list)))
            
            train_category_list = train_df['category'].unique().tolist()
            test_category_list = test_df['category'].unique().tolist()
            category_encoder = LabelBinarizer()
            category_encoder.fit(list(set(train_category_list + test_category_list)))

            train_data_loader, val_data_loader = get_train_val_loaders(train_data_path=train_data_path, \
                                                        val_data_path=val_data_path, \
                                                        host_encoder=host_encoder, \
                                                        category_encoder=category_encoder, \
                                                        model_type=args.model_name, \
                                                        content=args.content, \
                                                        batch_size=args.batch_size, \
                                                        val_batch_size=args.valid_batch_size, \
                                                        num_workers=args.num_workers, \
                                                        augment=args.augment, \
                                                        extra_token=True)
            
        else:
        
            train_data_loader, val_data_loader = get_train_val_loaders(train_data_path=train_data_path, \
                                                        val_data_path=val_data_path, \
                                                        model_type=args.model_name, \
                                                        content=args.content, \
                                                        batch_size=args.batch_size, \
                                                        val_batch_size=args.valid_batch_size, \
                                                        num_workers=args.num_workers, \
                                                        augment=args.augment, \
                                                        extra_token=False)
    else:
        raise NotImplementedError

    # start training
    training(
            args.content, \
            args.n_splits, \
            args.fold, \
            train_data_loader, \
            val_data_loader, \
            args.model_type, \
            args.model_name, \
            args.hidden_layers, \
            args.optimizer, \
            args.lr_scheduler, \
            args.lr, \
            args.warmup_proportion, \
            args.batch_size, \
            args.valid_batch_size, \
            args.num_epoch, \
            args.start_epoch, \
            args.accumulation_steps, \
            args.checkpoint_folder, \
            args.load_pretrain, \
            args.seed, \
            args.loss, \
            args.extra_token, \
            args.augment)

    gc.collect()
