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
from torch.utils.tensorboard import SummaryWriter

# import apex for mix precision training
from apex import amp

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
parser.add_argument('--optimizer', type=str, default='Ranger', required=False, help='specify the optimizer')
parser.add_argument("--lr_scheduler", type=str, default='CosineAnealing', required=False, help="specify the lr scheduler")
parser.add_argument("--lr", type=float, default=1e-4, required=False, help="specify the initial learning rate for training")
parser.add_argument("--batch_size", type=int, default=8, required=False, help="specify the batch size for training")
parser.add_argument("--valid_batch_size", type=int, default=32, required=False, help="specify the batch size for validating")
parser.add_argument("--num_epoch", type=int, default=15, required=False, help="specify the total epoch")
parser.add_argument("--accumulation_steps", type=int, default=4, required=False, help="specify the accumulation steps")
parser.add_argument('--num_workers', type=int, default=2, \
    required=False, help='specify the num_workers for testing dataloader')
parser.add_argument("--start_epoch", type=int, default=0, required=False, help="specify the start epoch for continue training")
parser.add_argument("--checkpoint_folder", type=str, default="/media/jionie/my_disk/Kaggle/Google_Quest_Answer/model", \
    required=False, help="specify the folder for checkpoint")
parser.add_argument('--load_pretrain', action='store_true', default=False, help='whether to load pretrain model')
parser.add_argument('--fold', type=int, default=0, required=True, help="specify the fold for training")
parser.add_argument('--seed', type=int, default=42, required=True, help="specify the seed for training")
parser.add_argument('--n_splits', type=int, default=5, required=True, help="specify the n_splits for training")


############################################################################## Define Constant
NUM_CLASS = 30


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
def training(fold,
            train_data_loader, 
            val_data_loader,
            model_type,
            model_name,
            optimizer_name,
            lr_scheduler_name,
            lr,
            batch_size,
            valid_batch_size,
            num_epoch,
            start_epoch,
            accumulation_steps,
            checkpoint_folder,
            load_pretrain,
            seed
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
    
    os.makedirs(checkpoint_folder + '/' + model_type + '/' + model_name, exist_ok=True)
    
    log = Logger()
    log.open(checkpoint_folder + '/' + model_type + '/' + model_name + '/' + 'fold_' + str(fold) + '_log_train.txt', mode='a+')
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
    

    ############################################################################### training parameters
    checkpoint_filename = model_type + '/' + model_name + '/' + 'fold_' + str(fold) + "_checkpoint.pth"
    checkpoint_filepath = os.path.join(checkpoint_folder, checkpoint_filename)


    ############################################################################### model
    if model_type == "bert":
        model = QuestNet(model_type=model_name, n_classes=NUM_CLASS)
    else:
        raise NotImplementedError

    model = model.cuda()
    
    if load_pretrain:
        load(model, checkpoint_filepath)

    ############################################################################### optimizer
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif optimizer_name == "Ranger":
        optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()), lr, weight_decay=1e-5)
    else:
        raise NotImplementedError
    
    ############################################################################### lr_scheduler
    if lr_scheduler_name == "CosineAnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-5, last_epoch=-1)
        lr_scheduler_each_iter = False
    elif lr_scheduler_name == "WarmRestart":
        scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-6)
        lr_scheduler_each_iter = False
    else:
        raise NotImplementedError

    log.write('net\n  %s\n'%(model_name))
    log.write('optimizer\n  %s\n'%(optimizer_name))
    log.write('schduler\n  %s\n'%(lr_scheduler_name))
    log.write('\n')

    ###############################################################################  mix precision
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

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
    valid_metric_optimal = np.inf
    
    # define tensorboard writer and timer
    writer = SummaryWriter()
    start_timer = timer()
    
    # define criterion
    criterion = nn.BCEWithLogitsLoss()
    # criterion = FocalLoss()
    
    for epoch in range(1, num_epoch+1):

        # init in-epoch statistics
        labels_train = None
        pred_train   = None
        labels_val   = None
        pred_val     = None
        
        # update lr and start from start_epoch  
        if (not lr_scheduler_each_iter):
            if epoch < 8:
                if epoch != 1:
                    scheduler.step()
            else:
                optimizer.param_groups[0]['lr'] = 1e-5

        # if (not lr_scheduler_each_iter):
        #     if epoch < 11:
        #         if epoch != 1:
        #             scheduler.step()
        #             # scheduler = warm_restart(scheduler, T_mult=2) 
        #     elif epoch < 21:
        #         optimizer.param_groups[0]['lr'] = 1e-4
        #     else:
        #         optimizer.param_groups[0]['lr'] = 1e-5
                
        # affect_rate = CosineAnnealingWarmUpRestarts(epoch, T_0=num_epoch, T_warmup=5, gamma=0.8,)
        # optimizer.param_groups[0]['lr'] = affect_rate * lr
        
        # if epoch < 5:
        #     optimizer.param_groups[0]['lr'] = affect_rate * lr
        # elif epoch < 10:
        #     lr = 1e-4
        #     optimizer.param_groups[0]['lr'] = affect_rate * lr
        # elif epoch < 20:
        #     optimizer.param_groups[0]['lr'] = 5e-5
        # else:
        #     optimizer.param_groups[0]['lr'] = 1e-5 
           
        if (epoch < start_epoch):
            continue
        
        log.write("Epoch%s\n" % epoch)
        log.write('\n')
            
        for param_group in optimizer.param_groups:
            rate = param_group['lr']

        sum_train_loss = np.zeros_like(train_loss)
        sum_train = np.zeros_like(train_loss)

        # init optimizer
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        
        for tr_batch_i, (token_ids, seg_ids, labels) in enumerate(train_data_loader):
            
            # set model training mode
            model.train() 

            # set input to cuda mode
            token_ids = token_ids.cuda()
            seg_ids   = seg_ids.cuda()
            labels    = labels.cuda().float()

            # predict and calculate loss (only need torch.sigmoid when inference)
            prediction = model(token_ids, seg_ids)  
            loss = criterion(prediction, labels)

            # adjust lr
            if (lr_scheduler_each_iter):
                scheduler.step(tr_batch_i)

            # use apex
            with amp.scale_loss(loss/accumulation_steps, optimizer) as scaled_loss:
                scaled_loss.backward()

            # don't use apex
            #loss.backward()
        
            if ((tr_batch_i+1) % accumulation_steps == 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()

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
                valid_metric = []
                
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

        val_metric_epoch = valid_loss[0]

        if (val_metric_epoch <= valid_metric_optimal):
            
            log.write('Validation metric improved ({:.6f} --> {:.6f}).  Saving model ...'.format(\
                    valid_metric_optimal, val_metric_epoch))

            valid_metric_optimal = val_metric_epoch
            torch.save(model.state_dict(), checkpoint_filepath)


if __name__ == "__main__":

    args = parser.parse_args()

    seed_everything(args.seed)

    # get train val split
    data_path = args.train_data_folder + "train.csv"
    get_train_val_split(data_path=data_path, \
                        save_path=args.train_data_folder, \
                        n_splits=args.n_splits, \
                        seed=args.seed)

    # get train_data_loader and val_data_loader
    train_data_path = args.train_data_folder + "split/train_fold_%s_seed_%s.csv"%(args.fold, args.seed)
    val_data_path   = args.train_data_folder + "split/val_fold_%s_seed_%s.csv"%(args.fold, args.seed)

    if args.model_type == "bert":
        train_data_loader, val_data_loader = get_train_val_loaders(train_data_path=train_data_path, \
                                                    val_data_path=val_data_path, \
                                                    model_type=args.model_name, \
                                                    batch_size=args.batch_size, \
                                                    val_batch_size=args.valid_batch_size, \
                                                    num_workers=args.num_workers)
    else:
        raise NotImplementedError

    # start training
    training(args.fold, \
            train_data_loader, \
            val_data_loader, \
            args.model_type, \
            args.model_name, \
            args.optimizer, \
            args.lr_scheduler, \
            args.lr, \
            args.batch_size, \
            args.valid_batch_size, \
            args.num_epoch, \
            args.start_epoch, \
            args.accumulation_steps, \
            args.checkpoint_folder, \
            args.load_pretrain, \
            args.seed)

    gc.collect()