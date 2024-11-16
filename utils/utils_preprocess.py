import os
import os.path as osp
import torch
from torch.utils.data import Dataset, DataLoader

def get_train_test_readers(args, tokenizer):
    from data_loader import NewsDataset
    
    # Get train/test split
    filenames_train, filenames_test = getanchorinfo(args)
    
    # Create datasets
    trainset = NewsDataset(filenames_train, args, tokenizer, is_training=True)
    testset = NewsDataset(filenames_test, args, tokenizer, is_training=False)
    
    # Create dataloaders
    train_reader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)
    valid_reader = DataLoader(testset, batch_size=args.valid_batch_size, shuffle=False)
    
    return train_reader, valid_reader

def getanchorinfo(args):
    # This is a placeholder function - in a real implementation, this would load
    # and process the anchor information from the dataset
    train_files = []
    test_files = []
    
    data_dir = osp.join(args.root, args.dataset, 'raw')
    all_files = os.listdir(data_dir)
    
    # Simple split based on test_size
    split_idx = int(len(all_files) * (1 - args.test_size))
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]
    
    return train_files, test_files
