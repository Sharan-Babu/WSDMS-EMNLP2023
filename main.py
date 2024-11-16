import argparse
import configparser
import datetime
import logging
import math
import os
import os.path as osp
import random
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW

sys.path.append("..")

from HotCakeModels import HotCakeForSequenceClassification
from bert_model import BertForSequenceEncoder
from analysis_results import analyze_results
from utils.utils_misc import get_eval_report, print_results, set_args_from_config, save_results_to_tsv
from utils.utils_preprocess import get_train_test_readers, getanchorinfo
from data_loader import get_datasets

logger = logging.getLogger(__name__)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def correct_prediction(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct


def cuda_memory_info():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    print(f'total    : {t}')
    print(f'free     : {f}')
    print(f'used     : {a}')


def eval_model(model, validset_reader, results_eval=None, args=None, epoch=0, writer=None, counters_test=None):
    model.eval()
    correct_pred = 0.0
    preds_all, labs_all, logits_all, filenames_test_all = [], [], [], []

    # get basice info
    for index, data in enumerate(validset_reader):
        inputs, lab_tensor, filenames_test, aux_info = data

        prob = model(inputs, aux_info)

        correct_pred += correct_prediction(prob, lab_tensor)
        preds_all += prob.max(1)[1].tolist()
        logits_all += prob.tolist()
        labs_all += lab_tensor.tolist()
        filenames_test_all += filenames_test

    preds_np = np.array(preds_all)  
    labs_np = np.array(labs_all)  
    logits_np = np.array(logits_all)

    # get results and analysis
    if counters_test is not None:
        analyze_results(labs_np, preds_np, counters_test, filenames_test_all, epoch, args)

    results = get_eval_report(labs_np, preds_np)
    print_results(results, epoch, args=args, dataset_split_name="Eval")
    if results_eval is not None:
        results_eval[epoch] = results
    dev_accuracy = correct_pred / validset_reader.total_num

    # write results into file
    if writer is not None:
        writer.add_pr_curve('pr_curve', labels=labs_np, predictions=np.exp(logits_np)[:, 1], global_step=epoch)
        writer.add_scalar("Acc/Test", dev_accuracy, global_step=epoch)

    return dev_accuracy



def train_model(model, args, train_loader, test_loader, writer, experiment_name):
    # Calculate total training steps
    total_steps = len(train_loader) * args.num_train_epochs
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=float(args.learning_rate))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_proportion),
        num_training_steps=total_steps
    )
    
    # Initialize training variables
    best_accuracy = 0.0
    best_epoch = 0
    global_step = 0
    tr_loss = 0.0
    
    # Training loop
    for epoch in range(int(args.num_train_epochs)):
        model.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            # Forward pass
            outputs = model(batch, None)
            loss = F.nll_loss(outputs, batch['label'])
            
            # Backward pass
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if writer:
                    writer.add_scalar('train/loss', epoch_loss, global_step)
        
        # Evaluation
        model.eval()
        eval_loss = 0.0
        eval_accuracy = 0.0
        eval_steps = 0
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(batch, None)
                tmp_eval_loss = F.nll_loss(outputs, batch['label'])
                eval_loss += tmp_eval_loss.item()
                
                preds = outputs.argmax(dim=1)
                eval_accuracy += (preds == batch['label']).float().mean().item()
                eval_steps += 1
        
        eval_loss = eval_loss / eval_steps
        eval_accuracy = eval_accuracy / eval_steps
        
        if writer:
            writer.add_scalar('eval/loss', eval_loss, epoch)
            writer.add_scalar('eval/accuracy', eval_accuracy, epoch)
        
        logger.info(f'Epoch {epoch+1}/{args.num_train_epochs}')
        logger.info(f'Average loss: {epoch_loss:.4f}')
        logger.info(f'Eval loss: {eval_loss:.4f}')
        logger.info(f'Eval accuracy: {eval_accuracy:.4f}')
        
        # Save best model
        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            best_epoch = epoch
            
            # Save model
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)
            
            model_path = os.path.join(args.outdir, f'best_model_{experiment_name}.pt')
            torch.save(model.state_dict(), model_path)
            
            logger.info(f'New best accuracy: {best_accuracy:.4f}')
            logger.info(f'Saved model to {model_path}')
        
        # Early stopping
        if epoch - best_epoch >= args.patience:
            logger.info(f'Early stopping triggered. Best accuracy: {best_accuracy:.4f} at epoch {best_epoch}')
            break
    
    return best_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patience', type=int, default=20, help='Patience')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--train_path', help='train path')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=2)

    parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')

    parser.add_argument("--eval_step", default=500, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "truncated and padded are turned on.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 of training")
    # ------------------------------------------
    # Additional args
    # ------------------------------------------

    parser.add_argument("--seed", default=21, type=int,
                        help="Random state")
    parser.add_argument("--kernel", default=21, type=int,
                        help="Number of kernels")
    parser.add_argument("--lambda", default=1e-1, type=float,
                        help="lambda value used")
    parser.add_argument("--root", default='../Demo', type=str,
                        help="")
    parser.add_argument("--kfold_index", default=-1, type=int,
                        help="Run this for K-fold cross validation")

    parser.add_argument('--debug', action='store_true', help='Debug')

    parser.add_argument("--mode", type=str, default="HotCake", help="Model mode: HotCake, HotCake-SW, or HotCake-FC")
    parser.add_argument("--sample_suffix", type=str, default="", help="Suffix for sample identification")
    parser.add_argument("--sigma", type=float, default=1.0, help="Sigma value for kernel")
    parser.add_argument("--dataset", type=str, default="politifact", help="Dataset to use: politifact, gossipcop, or buzznews")
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size ratio")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Validation batch size")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--enable_tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--bert_pretrain", type=str, default="bert-base-cased", help="Pretrained BERT model to use")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # ------------------------------------------
    # Config
    # ------------------------------------------

    config = configparser.ConfigParser(allow_no_value=True)

    args = parser.parse_args()

    # # train on DLTS or ITP
    # if args.itp:
    #     os.chdir(osp.join("/home", "username", "KernelGAT", "kgat"))

    config = configparser.ConfigParser()
    config.read(osp.join("config", args.config_file))

    args = set_args_from_config(args, config)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    timestr = datetime.datetime.now().strftime("%m-%d_%H")

    if args.dataset == 'politifact':
        experiment_name = f"{'PolitiFact'}_{timestr}_{args.mode}_batch{args.train_batch_size}_{args.num_train_epochs}_lr{args.learning_rate}_K{args.kernel}_Sig{args.sigma}{'_KFold' + str(args.kfold_index) if args.kfold_index >= 0 else ''}_{args.sample_suffix}"
    elif args.dataset == 'gossipcop':
        experiment_name = f"{'GossipCop'}_{timestr}_{args.mode}_batch{args.train_batch_size}_{args.num_train_epochs}_lr{args.learning_rate}_K{args.kernel}_Sig{args.sigma}{'_KFold' + str(args.kfold_index) if args.kfold_index >= 0 else ''}_{args.sample_suffix}"
    else:
        experiment_name = f"{'BuzzNews'}_{timestr}_{args.mode}_batch{args.train_batch_size}_{args.num_train_epochs}_lr{args.learning_rate}_K{args.kernel}_Sig{args.sigma}{'_KFold' + str(args.kfold_index) if args.kfold_index >= 0 else ''}_{args.sample_suffix}"

    if not osp.exists(osp.join("..", "logs", experiment_name)):
        os.mkdir(osp.join("..", "logs", experiment_name))

    handlers = [logging.FileHandler(osp.join("..", "logs", experiment_name, f"log_{experiment_name}.txt")),
                logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logger.info(f'{args.dataset} Start training!')
    logger.info(f'Using batch size {args.train_batch_size} | accumulation {args.gradient_accumulation_steps}')

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
    label_map = {
        'real': 0,
        'fake': 1
    }
    train_dataset, test_dataset = get_datasets(args)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False
    )
    
    # Initialize model
    logger.info("Initializing BERT model")
    model = HotCakeForSequenceClassification.from_pretrained(args.bert_pretrain)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=float(args.learning_rate))
    
    # Initialize tensorboard
    if args.enable_tensorboard:
        logger.info("Initializing Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    else:
        writer = None
    
    # Train the model
    best_accuracy = train_model(model, args, train_loader, test_loader, writer, "politifact")
    
    logger.info(f"Best accuracy: {best_accuracy}")
