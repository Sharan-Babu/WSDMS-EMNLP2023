import re
import os
import numpy as np
import torch
from torch.autograd import Variable
import json
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import random


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence one token at a time. 
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def truncate_seq_size(inp_padding_li, msk_padding_li, seg_padding_li, max_seq_size, max_seq_length):
    if max_seq_size != -1:
        inp_padding = inp_padding_li[:max_seq_size]
        msk_padding = msk_padding_li[:max_seq_size]
        seg_padding = seg_padding_li[:max_seq_size]
        inp_padding += ([[0] * max_seq_length] * (max_seq_size - len(inp_padding)))
        msk_padding += ([[0] * max_seq_length] * (max_seq_size - len(msk_padding)))
        seg_padding += ([[0] * max_seq_length] * (max_seq_size - len(seg_padding)))
    return inp_padding, msk_padding, seg_padding


def tok2int_sent(sentence, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    sent_tmp = ""
    if isinstance(sentence, list) and len(sentence) == 3:
        sent_a, title, sent_b = sentence
    elif isinstance(sentence, str):
        sent_a, title, sent_b = sentence, None, None
    else:
        raise Exception("Error: Fake News data format is incorrect")
    tokens_a = tokenizer.tokenize(sent_a)

    tokens_b = None
    tokens_t = None
    if sent_b and title:
        tokens_t = tokenizer.tokenize(title)
        tokens_b = tokenizer.tokenize(sent_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4 - len(tokens_t))
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    if tokens_b and tokens_t:
        tokens = tokens + tokens_t + ["[SEP]"] + tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + len(tokens_t) + 2)
    # print (tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def tok2int_sent_str(sentence, tokenizer, max_seq_length, format="str"):
    '''
    Convert all evidences for one example
    '''
    input_ids_li, input_mask_li, input_seg_li = [], [], []

    # claim, evidence are in the format of paragraph
    max_sent_length = max_seq_length - 2
    if format == "str":
        assert isinstance(sentence, str)
        sent_a = sentence

        # Account for [CLS] and [SEP] with "- 2"

        tokens_a = tokenizer.tokenize(sent_a)
        n_sents = int(np.ceil(len(tokens_a) / max_sent_length))
        tokens_a_li = [tokens_a[i * max_sent_length:(i + 1) * max_sent_length] for i in range(n_sents)]

    # claim, evidence(posts)
    elif format == "list":
        assert isinstance(sentence, list)
        tokens_a_li, tokens_title_li, tokens_b_li = [], [], []
        for tup in sentence:
            if len(tup) == 3:
                sent_a, title, sent_b = tup



            elif len(tup) == 2:
                sent_a, sent_b = tup
                title = ""
            if isinstance(sent_b, list):
                sent_b = ' '.join(sent_b)
            sent_a, title, sent_b = str(sent_a), str(title), str(sent_b)

            tokens_a = tokenizer.tokenize(sent_a)
            tokens_t = tokenizer.tokenize(title)
            tokens_b = tokenizer.tokenize(sent_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4 - len(tokens_t))
            tokens_a_li += [tokens_a]
            tokens_b_li += [tokens_b]
            tokens_title_li += [tokens_t]

    elif format == "twt":
        tokens_a_li = [tokenizer.tokenize(str(sent_a))[:max_sent_length] for sent_a in sentence]
    else:
        raise NotImplementedError

    def add_paddings(segment_ids, input_ids_li, input_mask_li, input_seg_li):
        # nonlocal input_ids_li, input_mask_li, input_seg_li
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        input_ids_li += [input_ids]
        input_mask_li += [input_mask]
        input_seg_li += [segment_ids]

    if format == "str":
        for tokens_a in tokens_a_li:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            add_paddings(segment_ids, input_ids_li, input_mask_li, input_seg_li)
    elif format == "list":
        for tokens_a, tokens_t, tokens_b in zip(tokens_a_li, tokens_title_li, tokens_b_li):
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_t != []:
                tokens += tokens_t + ["[SEP]"]
                segment_ids += [1] * (len(tokens_t) + 1)

            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

            add_paddings(segment_ids, input_ids_li, input_mask_li, input_seg_li)

    elif format == "twt":
        for tokens_a in tokens_a_li:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            add_paddings(segment_ids, input_ids_li, input_mask_li, input_seg_li)
    else:
        raise NotImplementedError

    return input_ids_li, input_mask_li, input_seg_li


def tok2int_list_claim_evi_pair(claim_evi_pairs, filename, labels_d, tokenizer, max_seq_length, max_seq_size=-1):
    inp_padding_li, msk_padding_li, seg_padding_li = tok2int_sent_str(claim_evi_pairs, tokenizer, max_seq_length,
                                                                      format="list")

    inp_padding, msk_padding, seg_padding = truncate_seq_size(inp_padding_li, msk_padding_li, seg_padding_li,
                                                              max_seq_size, max_seq_length)
    return inp_padding, msk_padding, seg_padding


def tok2int_list_case_study(example, tokenizer, max_seq_length, max_seq_size=-1):
    input_ids_li_p, input_mask_li_p, input_seg_li_p = tok2int_sent_str(example, tokenizer, max_seq_length,
                                                                       format="list")
    inp_padding, msk_padding, seg_padding = truncate_seq_size(input_ids_li_p, input_mask_li_p, input_seg_li_p,
                                                              max_seq_size, max_seq_length)
    return inp_padding, msk_padding, seg_padding


def tok2int_list_concat(example, filename, labels_d, tokenizer, max_seq_length, max_seq_size=-1, keep_claim=True,
                        only_claim=False):
    '''
    param: example consists of:
        example[0]: claim-posts.
        example[1]: news sentences. Claims(Sentences)
        example[2]: Label: 1 for Fake and 0 for Real
    '''

    if not only_claim:
        input_ids_li_p, input_mask_li_p, input_seg_li_p = tok2int_sent_str(example[0], tokenizer, max_seq_length,
                                                                           format="list")
    if only_claim or keep_claim:
        if isinstance(example[1], list):
            label = example[2]
            src_text = " ".join(example[1])
        elif isinstance(example, str):
            label = labels_d[filename]
            src_text = example
        input_ids_li_c, input_mask_li_c, input_seg_li_c = tok2int_sent_str(src_text, tokenizer, max_seq_length,
                                                                           format="str")

    if only_claim:
        inp_padding, msk_padding, seg_padding = truncate_seq_size(input_ids_li_c,
                                                                  input_mask_li_c,
                                                                  input_seg_li_c, max_seq_size,
                                                                  max_seq_length)

    elif keep_claim and not only_claim:

        src_text = " ".join(example[1])
        input_ids_li_c, input_mask_li_c, input_seg_li_c = tok2int_sent_str(src_text, tokenizer, max_seq_length,
                                                                           format="str")

        inp_padding, msk_padding, seg_padding = truncate_seq_size(input_ids_li_c + input_ids_li_p,
                                                                  input_mask_li_c + input_mask_li_p,
                                                                  input_seg_li_c + input_seg_li_p, max_seq_size,
                                                                  max_seq_length)
    else:
        inp_padding, msk_padding, seg_padding = truncate_seq_size(input_ids_li_p, input_mask_li_p, input_seg_li_p,
                                                                  max_seq_size, max_seq_length)
    return inp_padding, msk_padding, seg_padding, label


def tok2int_twt(claim_evidence_pair, df, scores, tokenizer, max_seq_length, args, max_seq_size=-1):
    '''
    param: df is the pandas DataFrame that stores
        tweets, retweets, replies
    '''
    twt_text = df.text.to_list()

    evi_num_fi = 0
    inp_padding_li, msk_padding_li, seg_padding_li = [], [], []
    if claim_evidence_pair is not None:
        inp_padding_li, msk_padding_li, seg_padding_li = tok2int_sent_str(claim_evidence_pair, tokenizer,
                                                                          max_seq_length, format="list")

        evi_num_fi = len(inp_padding_li)

    def process_scores(scores, evi_num_fi, mode='linear'):

        scores_paddings = np.zeros((3, args.evi_num))

        scores = np.array(scores)
        if evi_num_fi >= args.evi_num:
            scores = scores[:, :args.evi_num]
        elif evi_num_fi > 0:
            mean = (scores.sum(axis=1) / evi_num_fi).reshape(-1, 1)
            n_repeat = min(len(inp_padding_li) - evi_num_fi, args.evi_num - evi_num_fi)

            scores_twt = np.repeat(mean, repeats=n_repeat, axis=1)
            scores = np.concatenate((scores, scores_twt), axis=1)

        elif len(inp_padding_li) > 0:
            n_twt = len(inp_padding_li)
            scores = np.ones((3, n_twt)) / n_twt
            scores = scores[:, :args.evi_num]

        if np.count_nonzero(scores) > 0:
            if mode == 'log':
                scores = np.log(scores + 1)
            scores /= (scores.sum(axis=1)[:, np.newaxis])
            scores_paddings[:, :scores.shape[1]] = scores[:, :args.evi_num]

        # print(scores_paddings)
        return scores_paddings

    scores_paddings = process_scores(scores, evi_num_fi)

    inp_padding, msk_padding, seg_padding = truncate_seq_size(inp_padding_li, msk_padding_li, seg_padding_li,
                                                              max_seq_size, max_seq_length)
    return inp_padding, msk_padding, seg_padding, scores_paddings


def process_scores(scores, args, evi_num_fi, mode='linear'):
    if evi_num_fi >= args.evi_num:
        scores = [score_li[:args.evi_num] for score_li in scores]
    else:
        scores += [0] * (args.evi_num - evi_num_fi)
    scores = np.array(scores)
    if np.count_nonzero(scores) > 0:
        if mode == 'log':
            scores = np.log(scores + 1)
        scores /= scores.sum()
    return scores.tolist()


def tok2int_list(src_list, tokenizer, max_seq_length, max_seq_size=-1):
    inp_padding = list()
    msk_padding = list()
    seg_padding = list()
    sent_tmp = ""
    for step, sent in enumerate(src_list):
        input_ids, input_mask, input_seg = tok2int_sent(sent_tmp + sent, tokenizer, max_seq_length)
        inp_padding.append(input_ids)
        msk_padding.append(input_mask)
        seg_padding.append(input_seg)

    if max_seq_size != -1:
        inp_padding = inp_padding[:max_seq_size]
        msk_padding = msk_padding[:max_seq_size]
        seg_padding = seg_padding[:max_seq_size]
        inp_padding += ([[0] * max_seq_length] * (max_seq_size - len(inp_padding)))
        msk_padding += ([[0] * max_seq_length] * (max_seq_size - len(msk_padding)))
        seg_padding += ([[0] * max_seq_length] * (max_seq_size - len(seg_padding)))
    return inp_padding, msk_padding, seg_padding


class NewsDataset(torch.utils.data.Dataset):
    ''' For data iteration '''
    def __init__(self, filenames, root_dir, tokenizer, max_len=130):
        self.filenames = filenames
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load labels
        self.labels = torch.load(os.path.join(root_dir, 'raw', 'politifact_labels.pt'))
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Load article data
        with open(os.path.join(self.root_dir, 'raw', filename), 'r', encoding='utf-8') as f:
            article = json.load(f)
        
        # Get text and label
        claim = article['claim']
        evidence = article['evidence'][0]  # Use first evidence
        label = self.labels[filename]
        
        # Tokenize
        encoded = self.tokenizer(
            claim,
            evidence,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        # Convert to tensors
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        token_type_ids = encoded['token_type_ids'].squeeze(0)
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': label
        }

def get_train_test_split(args):
    """Get train and test filenames with stratified split."""
    labels_d = torch.load(os.path.join(args.root, 'raw', f"{args.dataset}_labels.pt"))

    filenames_real = [x for x, y in labels_d.items() if y == 0]
    filenames_fake = [x for x, y in labels_d.items() if y == 1]

    random.shuffle(filenames_real)
    random.shuffle(filenames_fake)

    n_train_real = int(len(filenames_real) * (1-args.test_size))
    n_train_fake = int(len(filenames_fake) * (1-args.test_size))

    # Note: split should be fixed
    filenames_train = filenames_real[:n_train_real] + filenames_fake[:n_train_fake]
    filenames_test = filenames_real[n_train_real:] + filenames_fake[n_train_fake:]
    return filenames_train, filenames_test

def get_datasets(args):
    """Create train and test datasets"""
    # Get train/test split
    filenames_train, filenames_test = get_train_test_split(args)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain)
    
    # Create datasets
    train_dataset = NewsDataset(filenames_train, args.root, tokenizer, max_len=args.max_len)
    test_dataset = NewsDataset(filenames_test, args.root, tokenizer, max_len=args.max_len)
    
    return train_dataset, test_dataset


class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, label_map, tokenizer, args, inputs, filenames, labels, test=False,
                 batch_size=64):
        self.cuda = args.cuda

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.args = args
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        self.sent_num = args.sent_num
        self.min_evi_num = args.min_evi_num
        self.label_map = label_map

        self.inputs = inputs
        self.filenames = filenames
        self.labels = labels
        self.test = test

        # Random shuffling
        self.total_num = len(inputs)
        if self.test:
            self.total_step = int(self.total_num / batch_size)  # np.ceil(self.total_num * 1.0 / batch_size)
        else:
            self.total_step = int(self.total_num / batch_size)
        # Drop last by default
        self.total_num = self.total_step * batch_size
        self.step = 0

    def process_sent(self, sentence):
        sentence = re.sub(" LSB.*?RSB", "", sentence)
        sentence = re.sub("LRB RRB ", "", sentence)
        sentence = re.sub("LRB", " ( ", sentence)
        sentence = re.sub("RRB", " )", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)
        sentence = re.sub("http://", "", sentence)
        sentence = re.sub("@", "", sentence)
        sentence = re.sub("#", "", sentence)


        return sentence

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def to_tensors(self, inp_paddings, msk_paddings, seg_paddings):
        inp_tensor = Variable(
            torch.LongTensor(inp_paddings)).view(-1, self.evi_num, self.max_len)
        msk_tensor = Variable(
            torch.LongTensor(msk_paddings)).view(-1, self.evi_num, self.max_len)
        seg_tensor = Variable(
            torch.LongTensor(seg_paddings)).view(-1, self.evi_num, self.max_len)

        if self.cuda:
            inp_tensor = inp_tensor.cuda()
            msk_tensor = msk_tensor.cuda()
            seg_tensor = seg_tensor.cuda()

        return inp_tensor, msk_tensor, seg_tensor

    def load_inputs(self):
        """
        Load inputs
        """
        inputs = self.inputs[self.step * self.batch_size: (self.step + 1) * self.batch_size]
        labels = self.labels[self.step * self.batch_size: (self.step + 1) * self.batch_size]
        inp_paddings, msk_paddings, seg_paddings = [], [], []
        for step in range(len(inputs)):
            inp, msk, seg = inputs[step]
            inp_paddings += inp
            msk_paddings += msk
            seg_paddings += seg
        lab_tensor = Variable(torch.LongTensor(labels))
        if self.cuda:
            lab_tensor = lab_tensor.cuda()
        return self.to_tensors(inp_paddings, msk_paddings, seg_paddings), lab_tensor

    def load_filenames(self):
        filenames = self.filenames[self.step * self.batch_size: (self.step + 1) * self.batch_size]
        return filenames

    def next(self):
        if self.step < self.total_step:
            (inp_tensor, msk_tensor, seg_tensor), lab_tensor = self.load_inputs()
            filenames = self.load_filenames()

            self.step += 1
            return (inp_tensor, msk_tensor, seg_tensor, self.step - 1), lab_tensor, \
                   filenames, aux_info
        else:
            self.step = 0
            raise StopIteration()
