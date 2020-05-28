import os
import time
import json
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn, optim
from argparse import ArgumentParser
from easydict import EasyDict as edict
from torch.nn.init import xavier_uniform_
from misc.HybridCNNLong import HybridCNNLong
from torch.distributions.categorical import Categorical
from text_datasets import get_quora_texts_pretrained_vocab
from utils import to_one_hot, QUORA_PARAPHRASE_PRETRAINED_DEFAULT_CONFIG_PATH

def mask_in_place_and_calc_length(texts, text_log_probs, end_token, pad_token=0):
    t4 = (texts == end_token).cumsum(dim=1).bool()
    mask = torch.cat((torch.zeros(t4.size(0), 1, dtype=torch.bool, device=t4.device), t4[:, :-1]), dim=1)
    text_lens = mask.size(1) - mask.sum(dim=1)
    texts.masked_fill_(mask, pad_token)
    text_log_probs.masked_fill_(mask.unsqueeze(2), pad_token)
    return texts, text_log_probs, text_lens

class TextGenerator(nn.Module):
    """Load weights from faster_text_gen_v1.pth"""

    def __init__(self, d_vocab, d_text_feature, d_gen_hidden, d_max_seq_len, d_gen_layers=1, gen_dropout=0, pad_token=0, start_token=-1, end_token=-1):
        super().__init__()

        assert start_token >= 0 and start_token < d_vocab
        assert end_token >= 0 and end_token < d_vocab
        assert pad_token >= 0 and pad_token < d_vocab

        self.d_vocab = d_vocab
        self.pad_token = pad_token
        self.end_token = end_token
        self.start_token = start_token
        self.gen_dropout = gen_dropout
        self.d_gen_layers = d_gen_layers
        self.d_gen_hidden = d_gen_hidden
        self.d_max_seq_len = d_max_seq_len
        self.d_text_feature = d_text_feature

        self.define_module()

    def define_module(self):
        self.embed = nn.Embedding(self.d_vocab, self.d_text_feature)
        self.rnn_cell = nn.LSTM(self.d_text_feature, self.d_gen_hidden, self.d_gen_layers, dropout=(0 if self.d_gen_layers == 1 else self.gen_dropout), batch_first=True)
        self.drop = nn.Dropout(self.gen_dropout)
        self.fc_logits = nn.Linear(self.d_gen_hidden, self.d_vocab)
        self.log_soft = nn.LogSoftmax(dim=-1)

    def step(self, xs, hs):
        xs, hs = self.rnn_cell(xs, hx=hs)
        log_probs = self.log_soft(self.fc_logits(self.drop(xs)))
        return log_probs, hs # (d_batch, 1, d_vocab), ((1, d_batch, d_gen_hidden), (1, d_batch, d_gen_hidden))

    def forward(self, text_features):
        '''
        encoded : (batch_size, feat_size)
        seq: (batch_size, seq_len)
        lengths: (batch_size, )
        '''
        d_batch = text_features.size(0)
        device = text_features.device
        d_max_seq_len = self.d_max_seq_len

        text_log_probs = torch.zeros(d_batch, d_max_seq_len, self.d_vocab, device=device)
        text_log_probs[:, 0, :] = torch.log(to_one_hot(torch.tensor(self.start_token), self.d_vocab))[None, :]
        texts = torch.full((d_batch, d_max_seq_len), self.pad_token, dtype=torch.long, device=device)
        texts[:, 0] = self.start_token
        hs = None

        for i in range(1, d_max_seq_len):
            if i > 1:
                word_embeddings = self.embed(words) # (d_batch, 1) -> (d_batch, 1, d_text_feature)
            else:
                word_embeddings = text_features.unsqueeze(1) # (d_batch, d_text_feature) -> (d_batch, 1, d_text_feature)
            log_probs, hs = self.step(word_embeddings, hs)
            log_probs = log_probs.squeeze(1) # (d_batch, 1, d_vocab) -> (d_batch, 1, d_vocab)
            text_log_probs[:, i] = log_probs
            words = torch.multinomial(torch.exp(log_probs), 1)
            texts[:, i] = words.squeeze(1) # (d_batch, 1) -> (d_batch,)

        texts, text_log_probs, text_lens = mask_in_place_and_calc_length(texts, text_log_probs, self.end_token, self.pad_token)
        return texts, text_log_probs, text_lens

def read_and_update_config(config=None):
    defaults = dict(
        d_batch=100,# 256, # 8, # 16, # 32, # 64, # 128, # 256, # 512, OOM
        num_epochs=1,
        gen_beta1=0.5,
        dis_beta1=0.5,
        gen_lr=9.59e-5,
        dis_lr=9.38e-3,
        baseline_decay=0.08,
        discount_factor=0.23,
        gen_weight_decay=0.0,
        dis_weight_decay=1e-6,
        should_pad=True,
        pad_to_length=26, # 52,
        no_start_end=False,
        num_workers=0,
        num_fast_bleu_references=256,
        text_gen_weights_path='faster_text_gen_v1.pth',
        text_dis_weights_path='faster_text_dis_v1.pth',
    )
    if config is not None:
        if isinstance(config, str):
            assert os.path.isfile(config)
            with open(config, 'r') as f:
                config = json.load(f)
        else:
            assert isinstance(config, dict)
        defaults.update(config)
    return defaults

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-w', dest='weights_path', type=str, required=True, help='path to weights')
    parser.add_argument('-o', dest='output_dir', type=str, required=True, help='output folder name')
    parser.add_argument('-n', dest='num_samples', type=int, default=5000, help='number of samples to generate')
    parser.add_argument('--config', type=str, default=None, help='path to the config used to construct the model')
    return parser.parse_args()

def run():
    args = parse_args()
    output_dir = args.output_dir
    config = read_and_update_config(config=args.config)
    d_batch = config['d_batch']
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump(config, f)
    with open(f'{output_dir}/args.json', 'w') as f:
        json.dump(vars(args), f)
    print('running with args:')
    print(vars(args))
    print('running with config:')
    print(config)
    dataset, _ = get_quora_texts_pretrained_vocab(split='test', **config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('running on device:', device)
    print('constructing models:')
    quora_default_args = edict(torch.load(QUORA_PARAPHRASE_PRETRAINED_DEFAULT_CONFIG_PATH))
    assert quora_default_args.input_encoding_size == quora_default_args.txtSize
    d_text_feature = quora_default_args.txtSize
    dis_dropout = quora_default_args.drop_prob_lm
    d_dis_cnn = quora_default_args.cnn_dim
    d_gen_hidden = quora_default_args.rnn_size
    d_gen_layers = quora_default_args.rnn_layers
    gen_dropout = quora_default_args.drop_prob_lm
    text_gen = TextGenerator(d_vocab=dataset.d_vocab, d_text_feature=d_text_feature, d_gen_hidden=d_gen_hidden, d_max_seq_len=dataset.d_max_seq_len, d_gen_layers=d_gen_layers, gen_dropout=gen_dropout, pad_token=dataset.pad_token, start_token=dataset.start_token, end_token=dataset.end_token)
    text_gen.load_state_dict(torch.load(args.weights_path))
    text_gen.to(device).eval()
    print('generating samples:')
    end_token = text_gen.end_token
    def remove_padding_helper(xs):
        xs = xs[1:] # remove start_token
        for i, x in enumerate(xs):
            if x == end_token:
                return xs[:i] # remove end and pad tokens
        return xs
    def remove_padding(xss):
        if isinstance(xss, torch.Tensor):
            xss = xss.tolist()
        return [remove_padding_helper(xs) for xs in xss]
    fake_text_ints = []
    fake_text_strs = []
    num_generated = 0
    num_samples = args.num_samples
    with torch.no_grad():
        while num_generated < num_samples:
            noises = torch.randn(d_batch, d_text_feature, device=device)
            fake_texts = text_gen(noises)[0]
            fake_text_ints.extend(fake_texts.tolist())
            fake_text_strs.extend([dataset.decode_caption(fake_text) for fake_text in fake_texts.cpu()])
            num_generated += d_batch
    with open(f'{output_dir}/fake_texts.txt', 'w') as f:
        f.write('\n'.join(fake_text_strs))
    fake_text_ints = remove_padding(fake_text_ints)
    torch.save(fake_text_ints, f'{output_dir}/fake_texts.pth')
    with open(f'{output_dir}/fake_texts.txt', 'w') as f:
        f.write('\n'.join(fake_text_strs))
    print('DONE!')

if __name__ == '__main__':
    run()
