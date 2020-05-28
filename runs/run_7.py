# # Imports

# In[1]:


import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn, optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from easydict import EasyDict as edict
from torch.nn.init import xavier_uniform_
from misc.HybridCNNLong import HybridCNNLong
from nltk.translate.bleu_score import sentence_bleu
from torch.distributions.categorical import Categorical
from text_datasets import get_emnlp_2017_news_combined_vocab, get_emnlp_2017_news_pretrained_vocab, get_quora_texts_pretrained_vocab
from utils import GLOVE_5102_300_PATH, to_one_hot, QUORA_PARAPHRASE_PRETRAINED_PATH, QUORA_PARAPHRASE_PRETRAINED_CHECKPOINT_PATH, QUORA_PARAPHRASE_PRETRAINED_WORD_ID_TO_WORD_PATH, QUORA_PARAPHRASE_PRETRAINED_SPECIAL_TOKENS_PATH, QUORA_PARAPHRASE_PRETRAINED_DEFAULT_CONFIG_PATH, QUORA_TEXT_SPLITS_PATH, QUORA_TEXT_PRETRAINED_VOCAB_VALID_SET_PATH, QUORA_TEXT_PRETRAINED_VOCAB_TEST_SET_PATH


# # Generator

# In[2]:


def mask_in_place_and_calc_length(texts, text_log_probs, end_token, pad_token=0):
    t4 = (texts == end_token).cumsum(dim=1).bool()
    mask = torch.cat((torch.zeros(t4.size(0), 1, dtype=torch.bool, device=t4.device), t4[:, :-1]), dim=1)
    text_lens = mask.size(1) - mask.sum(dim=1)
    texts.masked_fill_(mask, pad_token)
    mask = mask.unsqueeze(2).expand_as(text_log_probs)
    pad_mask = torch.zeros_like(mask)
    pad_mask[:, :, pad_token] = True
    text_log_probs.masked_fill_(mask, float('-inf'))
    text_log_probs.masked_fill_(mask & pad_mask, 0.0)
    return texts, text_log_probs, text_lens

def testing():
    d_batch = 2
    d_max_seq_len = 8
    d_vocab = 4
    pad_token = 0
    eos_token = 2

    texts = torch.randint(low=0, high=d_vocab, size=(d_batch, d_max_seq_len))
    print(texts)
    text_log_probs = F.log_softmax(torch.randn(d_batch, d_max_seq_len, d_vocab), dim=2)
    print(text_log_probs)
    texts, text_log_probs, text_lens = mask_in_place_and_calc_length(texts, text_log_probs, eos_token, pad_token)
    print('text_lens:', text_lens)
    print(texts)
    print(text_log_probs)

testing()


# # WIP Faster Generator

# In[3]:


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

def test_text_generator():
    d_batch = 4
    d_max_seq_len = 52
    d_text_feature = 512
    d_gen_hidden = 512
    gen_dropout = 0.5
    d_gen_layers = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#     dataset, _ = get_emnlp_2017_news_pretrained_vocab(d_batch=d_batch, should_pad=True, pad_to_length=d_max_seq_len)
    dataset, _ = get_quora_texts_pretrained_vocab(d_batch=d_batch, should_pad=True, pad_to_length=d_max_seq_len)

    text_gen = TextGenerator(d_vocab=dataset.d_vocab, d_text_feature=d_text_feature, d_gen_hidden=d_gen_hidden, d_max_seq_len=dataset.d_max_seq_len, d_gen_layers=d_gen_layers, gen_dropout=gen_dropout, pad_token=dataset.pad_token, start_token=dataset.start_token, end_token=dataset.end_token)
    text_gen.load_state_dict(torch.load('faster_text_gen_v1.pth'))
    text_gen.to(device).train()

    xs = torch.randn(d_batch, d_text_feature, device=device)

    texts, text_log_probs, text_lens = text_gen(xs)
    assert texts.size() == (d_batch, d_max_seq_len) and texts.dtype == torch.long
    assert text_log_probs.size() == (d_batch, d_max_seq_len, dataset.d_vocab) and text_log_probs.dtype == torch.float
    assert text_lens.size() == (d_batch,) and text_lens.dtype == torch.long

    for text in texts:
        print(dataset.decode_caption(text))
        print('-'*64)
    print('#'*64)
    for text in text_log_probs.argmax(dim=2):
        print(dataset.decode_caption(text))
        print('-'*64)

test_text_generator()


# # Discriminator

# In[4]:


def test_text_enc():
    d_batch = 4
    d_max_seq_len = 52

    quora_checkpoint = torch.load(QUORA_PARAPHRASE_PRETRAINED_CHECKPOINT_PATH)
    quora_default_args = torch.load(QUORA_PARAPHRASE_PRETRAINED_DEFAULT_CONFIG_PATH)
    quora_word_id_to_word = torch.load(QUORA_PARAPHRASE_PRETRAINED_WORD_ID_TO_WORD_PATH)
    word_id_to_word = quora_word_id_to_word
    d_vocab = len(word_id_to_word)

    args = edict(quora_default_args)
    d_text_feature = args.txtSize
    text_enc = HybridCNNLong(d_vocab, args.txtSize, dropout=args.drop_prob_lm, avg=1, cnn_dim=args.cnn_dim)
    text_enc.load_state_dict(quora_checkpoint['encoder_state_dict'])
    assert text_enc

    texts = torch.randn(d_batch, d_max_seq_len, d_vocab)
    text_features = text_enc(texts)
    assert text_features.size() == (d_batch, d_text_feature) and text_features.dtype == torch.float

test_text_enc()


# In[5]:


class TextDiscriminator(nn.Module):
    """An LSTM discriminator that operates on word indexes."""

    def __init__(self, d_vocab, d_text_feature, dis_dropout, d_dis_cnn, **kwargs):
        super().__init__()
        self.d_vocab = d_vocab
        self.d_text_feature = d_text_feature
        self.dis_dropout = dis_dropout
        self.d_dis_cnn = d_dis_cnn
        self.define_module()

    def define_module(self):
        self.text_enc = HybridCNNLong(self.d_vocab, self.d_text_feature, dropout=self.dis_dropout, avg=1, cnn_dim=self.d_dis_cnn)
        self.fc_logits = nn.Linear(self.d_text_feature, 1)

    def forward(self, sequence):
        text_features = self.text_enc(sequence)
        if torch.isnan(text_features).any():
            print('sequence:', sequence)
            print('text_features:', text_features)
            assert False
        return text_features, self.fc_logits(text_features).squeeze(1)

def test_text_disciminator():
    d_batch = 4
    d_max_seq_len = 52
    quora_default_args = torch.load(QUORA_PARAPHRASE_PRETRAINED_DEFAULT_CONFIG_PATH)
    quora_word_id_to_word = torch.load(QUORA_PARAPHRASE_PRETRAINED_WORD_ID_TO_WORD_PATH)
    word_id_to_word = quora_word_id_to_word
    d_vocab = len(word_id_to_word)
    args = edict(quora_default_args)
    d_text_feature = args.txtSize
    dis_dropout = args.drop_prob_lm
    d_dis_cnn = args.cnn_dim

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    text_dis = TextDiscriminator(d_vocab=d_vocab, d_text_feature=d_text_feature, dis_dropout=dis_dropout, d_dis_cnn=d_dis_cnn).to(device).train()
    text_dis.load_state_dict(torch.load('faster_text_dis_v1.pth'))
    assert text_dis

    texts = torch.randint(low=0, high=d_vocab, size=(d_batch, d_max_seq_len)).to(device)
    text_features, valids = text_dis(to_one_hot(texts, d_vocab))
    assert text_features.size() == (d_batch, d_text_feature) and text_features.dtype == torch.float
    assert valids.size() == (d_batch,) and valids.dtype == torch.float

    text_log_probs = -torch.rand(d_batch, d_max_seq_len, d_vocab).to(device)
    text_features, valids = text_dis(text_log_probs)
    assert text_features.size() == (d_batch, d_text_feature) and text_features.dtype == torch.float
    assert valids.size() == (d_batch,) and valids.dtype == torch.float

test_text_disciminator()


# # Training

# In[6]:

def pairwise_loss(xs, ys):
    return torch.mean(torch.clamp(torch.mm(xs, ys.t()) - torch.sum(xs*ys, dim=-1) + 1.0, min=0.0))

def train(text_gen, text_dis, dataset, loader, device, output_dir, references=None, text_gen_opt_weights_path=None, text_dis_opt_weights_path=None,
    d_batch=512,
    num_epochs=20,
    gen_beta1=0.5,
    dis_beta1=0.5,
    gen_lr=9.59e-5,
    dis_lr=9.38e-3,
    baseline_decay=0.08,
    discount_factor=0.23,
    gen_weight_decay=0.0,
    dis_weight_decay=1e-6,
    **kwargs):

    assert references is not None

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

    d_vocab = dataset.d_vocab
    d_max_seq_len = dataset.d_max_seq_len
    d_text_feature = text_dis.d_text_feature

    text_gen.train()
    text_dis.train()

    criterion = nn.BCEWithLogitsLoss()

    real_targets = torch.ones(d_batch, device=device)
    fake_targets = torch.zeros(d_batch, device=device)

    text_gen_opt = optim.Adam(text_gen.parameters())
    text_dis_opt = optim.Adam(text_dis.parameters())
#     text_gen_opt = optim.Adam(text_gen.parameters(), lr=gen_lr, betas=(gen_beta1, 0.999), weight_decay=gen_weight_decay)
#     text_dis_opt = optim.Adam(text_dis.parameters(), lr=dis_lr, betas=(dis_beta1, 0.999), weight_decay=dis_weight_decay)
    if text_gen_opt_weights_path:
        text_gen_opt.load_state_dict(text_gen_opt_weights_path)
    if text_dis_opt_weights_path:
        text_dis_opt.load_state_dict(text_dis_opt_weights_path)

    moving_avg = 0.0

    text_gen_losses = []
    text_dis_losses = []
    text_dis_real_scores = []
    text_dis_fake_scores = []

    fixed_noises = torch.randn(d_batch, d_text_feature, device=device)
    bleu_scores = {n:[] for n in range(1, 6)} # calculate BLEU 1 to 5 scores
    best_bleu_scores = {n: -1.0 for n in range(1, 6)}

    for epoch in range(1, num_epochs+1):
        print('#'*64)
        print('epoch:', epoch)
        epoch_text_gen_loss = 0.0
        epoch_text_dis_loss = 0.0
        epoch_avg = 0.0
        epoch_rew = 0.0
        epoch_real_score = 0.0
        epoch_fake_score = 0.0
        for i, batch in tqdm(enumerate(loader, start=1), total=len(loader)):
            real_texts, _ = batch
            real_texts = real_texts.to(device)
            noises = torch.randn(d_batch, d_text_feature, device=device)
            fake_texts, fake_text_log_probs, _ = text_gen(noises)

            # train discriminator
            text_dis.zero_grad()
            _, valids = text_dis(to_one_hot(real_texts, d_vocab))
            text_dis_real_loss = criterion(valids, real_targets)
            text_dis_real_loss.backward()
            epoch_real_score += valids.mean().item()

            _, valids = text_dis(torch.exp(fake_text_log_probs.detach()))
            text_dis_fake_loss = criterion(valids, fake_targets)
            text_dis_fake_loss.backward()
            epoch_fake_score += valids.mean().item()
            text_dis_opt.step()

            epoch_text_dis_loss += text_dis_real_loss.item() * 0.5 + text_dis_fake_loss.item() * 0.5

            # train generator
            text_gen.zero_grad()
            fake_text_features, valids = text_dis(torch.exp(fake_text_log_probs))
#             vloss = criterion(valids, real_targets)
#             ploss = pairwise_loss(noises, fake_text_features)
#             print('vloss:', vloss.item())
#             print('ploss:', ploss.item())
#             text_gen_loss = vloss + ploss
            text_gen_loss = criterion(valids, real_targets) + pairwise_loss(noises, fake_text_features)
            text_gen_loss.backward()
            text_gen_opt.step()

            epoch_text_gen_loss += text_gen_loss.item()

        epoch_text_gen_loss /= len(loader)
        epoch_text_dis_loss /= len(loader)
        epoch_real_score /= len(loader)
        epoch_fake_score /= len(loader)

        text_gen_losses.append(epoch_text_gen_loss)
        text_dis_losses.append(epoch_text_dis_loss)
        text_dis_real_scores.append(epoch_real_score)
        text_dis_fake_scores.append(epoch_fake_score)

        print('epoch_text_gen_loss:', epoch_text_gen_loss)
        print('epoch_text_dis_loss:', epoch_text_dis_loss)
        print('epoch_real_score:', epoch_real_score)
        print('epoch_fake_score:', epoch_fake_score)

        print('real texts')
        print('\n'.join([dataset.decode_caption(real_text) for real_text in real_texts.cpu()[:4]]))
        print('fake texts')
        with torch.no_grad():
            text_gen.eval()
            fake_texts = text_gen(noises)[0]
            text_gen.train()
        fake_text_strs = [dataset.decode_caption(fake_text) for fake_text in fake_texts.cpu()]
        print('\n'.join(fake_text_strs[:4]))
        with open(f'{output_dir}/epoch_{epoch}.txt', 'w') as f:
            f.write('\n'.join(fake_text_strs))

        # calculate BLEU 1 to 5
        hypos = remove_padding(fake_texts)
        for n in range(1, 6):
            weight = tuple(1. / n for _ in range(n))
            bleu_score = sum(sentence_bleu(references, hypo, weight) for hypo in hypos) / len(hypos) # smoothing_function=chencherry.method1)
            print(f'{n}-gram BLEU score:', bleu_score)
            bleu_scores[n].append(bleu_score)
            best = best_bleu_scores[n]
            if best < bleu_score:
                print('improved BLEU', n,'score from', best, 'to', bleu_score)
                best_bleu_scores[n] = bleu_score
                print('saving best models:')
                torch.save(text_gen.state_dict(), f'{output_dir}/bleu_{n}_text_gen.pth')
                torch.save(text_dis.state_dict(), f'{output_dir}/bleu_{n}_text_dis.pth')
                torch.save(text_gen_opt.state_dict(), f'{output_dir}/bleu_{n}_text_gen_opt.pth')
                torch.save(text_dis_opt.state_dict(), f'{output_dir}/bleu_{n}_text_dis_opt.pth')

        plt.close('all')
        plt.figure(figsize=(3*5, 4))
        xs = torch.arange(1, epoch+1)

        plt.subplot(1, 3, 1)
        plt.plot(xs, list(zip(text_gen_losses, text_dis_losses)))
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['text gen', 'text dis'])
        plt.title('losses')

        plt.subplot(1, 3, 2)
        plt.plot(xs, list(zip(text_dis_real_scores, text_dis_fake_scores)))
        plt.xlabel('epochs')
        plt.ylabel('logit/score')
        plt.legend(['real', 'fake'])
        plt.title('real and fake text scores')

        plt.subplot(1, 3, 3)
        plt.plot(xs, list(zip(*[bleu_scores[n] for n in range(1, 6)])))
        plt.xlabel('epochs')
        plt.ylabel('bleu')
        plt.legend([str(n) for n in range(1, 6)])
        plt.title('BLEU 1 to 5 scores')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics.png')
        plt.show()

        with open(f'{output_dir}/metrics.json', 'w') as f:
            json.dump(dict(text_gen_losses=text_gen_losses, text_dis_losses=text_dis_losses, text_dis_real_scores=text_dis_real_scores, text_dis_fake_scores=text_dis_fake_scores, bleu_scores=bleu_scores), f)

        print('saving latest models:')
        torch.save(text_gen.state_dict(), f'{output_dir}/latest_text_gen.pth')
        torch.save(text_dis.state_dict(), f'{output_dir}/latest_text_dis.pth')
        torch.save(text_gen_opt.state_dict(), f'{output_dir}/latest_text_gen_opt.pth')
        torch.save(text_dis_opt.state_dict(), f'{output_dir}/latest_text_dis_opt.pth')


# In[7]:


def read_and_update_config(config=None):
    defaults = dict(
        d_batch=256, # 8, # 16, # 32, # 64, # 128, # 256, # 512, OOM
        num_epochs=20,
#         gen_beta1=0.5,
#         dis_beta1=0.5,
#         gen_lr=9.59e-5,
#         dis_lr=9.38e-3,
#         baseline_decay=0.08,
#         discount_factor=0.23,
#         gen_weight_decay=0.0,
#         dis_weight_decay=1e-6,
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
            assert os.path.isfile(config_path)
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            assert isinstance(config, dict)
        defaults.update(config)
    return defaults


# In[8]:


def run(config, output_dir):
    config = read_and_update_config(config)
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump(config, f)
    print('running with config:')
    print(config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
    print('running on device:', device)

    print('loading dataset:')
#     dataset, loader = get_emnlp_2017_news_pretrained_vocab(**config)
    dataset, loader = get_quora_texts_pretrained_vocab(split='train', **config) # d_batch=d_batch, should_pad=True, pad_to_length=d_max_seq_len)
    print('d_max_seq_len:', dataset.d_max_seq_len)
    references = torch.load(QUORA_TEXT_PRETRAINED_VOCAB_VALID_SET_PATH)[:config['num_fast_bleu_references']]
    print('num validation set BLEU references:', len(references))

    print('constructing models:')
    args = edict(torch.load(QUORA_PARAPHRASE_PRETRAINED_DEFAULT_CONFIG_PATH))
    assert args.input_encoding_size == args.txtSize
    d_text_feature = args.txtSize
    dis_dropout = args.drop_prob_lm
    d_dis_cnn = args.cnn_dim
    d_gen_hidden = args.rnn_size
    d_gen_layers = args.rnn_layers
    gen_dropout = args.drop_prob_lm

    text_gen = TextGenerator(d_vocab=dataset.d_vocab, d_text_feature=d_text_feature, d_gen_hidden=d_gen_hidden, d_max_seq_len=dataset.d_max_seq_len, d_gen_layers=d_gen_layers, gen_dropout=gen_dropout, pad_token=dataset.pad_token, start_token=dataset.start_token, end_token=dataset.end_token)
    text_gen.load_state_dict(torch.load(config['text_gen_weights_path']))
    text_gen.to(device)

    text_dis = TextDiscriminator(d_vocab=dataset.d_vocab, d_text_feature=d_text_feature, dis_dropout=dis_dropout, d_dis_cnn=d_dis_cnn)
    text_dis.load_state_dict(torch.load(config['text_dis_weights_path']))
    text_dis.to(device)

    print('training:')
    train(text_gen, text_dis, dataset, loader, device, output_dir=output_dir, references=references, **config)
    print('finished training')


# # Runs

if __name__ == '__main__':
    run(config=dict(num_epochs=50), output_dir='run_7/')

# # End
