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
from text_decoder import TextDecoder
from text_encoder import TextEncoder
from easydict import EasyDict as edict
from torch.nn.init import xavier_uniform_
from misc.HybridCNNLong import HybridCNNLong
from latent_generator import LatentGenerator
from latent_discriminator import LatentDiscriminator
from nltk.translate.bleu_score import sentence_bleu
from torch.distributions.categorical import Categorical
from text_datasets import get_emnlp_2017_news_combined_vocab, get_emnlp_2017_news_pretrained_vocab, get_quora_texts_pretrained_vocab
from utils import GLOVE_5102_300_PATH, to_one_hot, QUORA_PARAPHRASE_PRETRAINED_PATH, QUORA_PARAPHRASE_PRETRAINED_CHECKPOINT_PATH, QUORA_PARAPHRASE_PRETRAINED_WORD_ID_TO_WORD_PATH, QUORA_PARAPHRASE_PRETRAINED_SPECIAL_TOKENS_PATH, QUORA_PARAPHRASE_PRETRAINED_DEFAULT_CONFIG_PATH, QUORA_TEXT_SPLITS_PATH, QUORA_TEXT_PRETRAINED_VOCAB_VALID_SET_PATH, QUORA_TEXT_PRETRAINED_VOCAB_TEST_SET_PATH


# # Training

# In[6]:


def train(lat_gen, text_dec, text_enc, lat_dis, dataset, loader, device, output_dir, references=None, text_dec_opt_weights_path=None, text_enc_opt_weights_path=None,
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

    end_token = dataset.end_token
    def remove_helper(xs):
        xs = xs[1:]
        for i, x in enumerate(xs):
            if x == end_token:
                return xs[:i]
        return xs
    def remove_special_tokens(xss):
        if isinstance(xss, torch.Tensor):
            xss = xss.tolist()
        return [remove_helper(xs) for xs in xss]

    d_vocab = dataset.d_vocab
    d_max_seq_len = dataset.d_max_seq_len
    d_text_feature = lat_gen.d_text_feature
    d_noise = lat_gen.d_noise

    text_enc.train()
    text_dec.train()
    lat_dis.train()
    lat_gen.train()

    criterion = nn.BCEWithLogitsLoss()

    real_targets = torch.ones(d_batch, device=device)
    fake_targets = torch.zeros(d_batch, device=device)

    text_dec_opt = optim.Adam(text_dec.parameters())
    text_enc_opt = optim.Adam(text_enc.parameters())
    if text_dec_opt_weights_path:
        text_dec_opt.load_state_dict(torch.load(text_dec_opt_weights_path))
    if text_enc_opt_weights_path:
        text_enc_opt.load_state_dict(torch.load(text_enc_opt_weights_path))

    moving_avg = 0.0

    text_dec_losses = []
    text_enc_losses = []
    text_enc_real_scores = []
    text_enc_fake_scores = []

    fixed_noises = torch.randn(d_batch, d_noise, device=device)
    bleu_scores = {n:[] for n in range(1, 6)} # calculate BLEU 1 to 5 scores
    best_bleu_scores = {n: -1.0 for n in range(1, 6)}

    batch_div_importance = 0.5

    for epoch in range(1, num_epochs+1):
        print('#'*64)
        print('epoch:', epoch)
        epoch_text_dec_loss = 0.0
        epoch_text_enc_loss = 0.0
        epoch_avg = 0.0
        epoch_rew = 0.0
        epoch_real_score = 0.0
        epoch_fake_score = 0.0
        for i, batch in tqdm(enumerate(loader, start=1), total=len(loader)):
            real_texts, _ = batch
            real_texts = real_texts.to(device)
            with torch.no_grad():
                fake_latents = lat_gen(torch.randn(d_batch, d_noise, device=device))
            fake_texts, fake_text_log_probs, _ = text_dec(fake_latents)

            # train discriminator
            text_enc.zero_grad()
            text_features = text_enc(to_one_hot(real_texts, d_vocab))
            valids = lat_dis(text_features)
            text_enc_real_loss = criterion(valids, real_targets)
            text_enc_real_loss.backward()
            epoch_real_score += valids.mean().item()

            text_features = text_enc(to_one_hot(fake_texts, d_vocab))
            valids = lat_dis(text_features)
            text_enc_fake_loss = criterion(valids, fake_targets)
            text_enc_fake_loss.backward()
            epoch_fake_score += valids.mean().item()
            text_enc_opt.step()

            epoch_text_enc_loss += text_enc_real_loss.item() * 0.5 + text_enc_fake_loss.item() * 0.5

            # train generator
            text_dec.zero_grad()
            action_log_probs = fake_text_log_probs.gather(dim=2, index=fake_texts.unsqueeze(2)).squeeze(2)
            seq_log_probs = action_log_probs.sum(dim=1)
            rewards = valids.detach()
            expected_reward = torch.mean(seq_log_probs * rewards)
            text_dec_loss = -expected_reward
            text_dec_loss.backward()
            text_dec_opt.step()

            epoch_text_dec_loss += text_dec_loss.item()

        epoch_text_dec_loss /= len(loader)
        epoch_text_enc_loss /= len(loader)
        epoch_real_score /= len(loader)
        epoch_fake_score /= len(loader)

        text_dec_losses.append(epoch_text_dec_loss)
        text_enc_losses.append(epoch_text_enc_loss)
        text_enc_real_scores.append(epoch_real_score)
        text_enc_fake_scores.append(epoch_fake_score)

        print('epoch_text_dec_loss:', epoch_text_dec_loss)
        print('epoch_text_enc_loss:', epoch_text_enc_loss)
        print('epoch_real_score:', epoch_real_score)
        print('epoch_fake_score:', epoch_fake_score)

        print('real texts')
        print('\n'.join([dataset.decode_caption(real_text) for real_text in real_texts.cpu()[:4]]))
        print('fake texts')
        with torch.no_grad():
            lat_gen.eval()
            text_dec.eval()
            fake_texts = text_dec(lat_gen(fixed_noises))[0]
            lat_gen.train()
            text_dec.train()
        fake_text_strs = [dataset.decode_caption(fake_text) for fake_text in fake_texts.cpu()]
        print('\n'.join(fake_text_strs[:4]))
        with open(f'{output_dir}/epoch_{epoch}.txt', 'w') as f:
            f.write('\n'.join(fake_text_strs))

        # calculate BLEU 1 to 5
        hypos = remove_special_tokens(fake_texts)
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
                torch.save(text_dec.state_dict(), f'{output_dir}/bleu_{n}_text_dec.pth')
                torch.save(text_enc.state_dict(), f'{output_dir}/bleu_{n}_text_enc.pth')
                torch.save(text_dec_opt.state_dict(), f'{output_dir}/bleu_{n}_text_dec_opt.pth')
                torch.save(text_enc_opt.state_dict(), f'{output_dir}/bleu_{n}_text_enc_opt.pth')

        plt.close('all')
        plt.figure(figsize=(3*5, 4))
        xs = torch.arange(1, epoch+1)

        plt.subplot(1, 3, 1)
        plt.plot(xs, list(zip(text_dec_losses, text_enc_losses)))
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['text gen', 'text dis'])
        plt.title('losses')

        plt.subplot(1, 3, 2)
        plt.plot(xs, list(zip(text_enc_real_scores, text_enc_fake_scores)))
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
            json.dump(dict(text_dec_losses=text_dec_losses, text_enc_losses=text_enc_losses, text_enc_real_scores=text_enc_real_scores, text_enc_fake_scores=text_enc_fake_scores, bleu_scores=bleu_scores), f)

        print('saving latest models:')
        torch.save(text_dec.state_dict(), f'{output_dir}/latest_text_dec.pth')
        torch.save(text_enc.state_dict(), f'{output_dir}/latest_text_enc.pth')
        torch.save(text_dec_opt.state_dict(), f'{output_dir}/latest_text_dec_opt.pth')
        torch.save(text_enc_opt.state_dict(), f'{output_dir}/latest_text_enc_opt.pth')

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

def run(config, output_dir):
    config = read_and_update_config(config)
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump(config, f)
    print('running with config:')
    print(config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('running on device:', device)

    print('loading dataset:')
#     dataset, loader = get_emnlp_2017_news_pretrained_vocab(**config)
    dataset, loader = get_quora_texts_pretrained_vocab(split='train', **config) # d_batch=d_batch, should_pad=True, pad_to_length=d_max_seq_len)
    print('d_max_seq_len:', dataset.d_max_seq_len)
    references = torch.load(QUORA_TEXT_PRETRAINED_VOCAB_VALID_SET_PATH)[:config['num_fast_bleu_references']]
    print('num validation set BLEU references:', len(references))

    print('constructing models:')
    d_batch = 512
    d_noise = 100
    d_vocab = 27699
    num_epochs = 50
    start_epoch = 1
    d_gen_layers = 1
    gen_dropout = 0.5
    d_max_seq_len = 26
    d_gen_hidden = 512
    d_dis_hidden = 512
    d_text_feature = 512
    d_text_enc_cnn = 512
    d_text_enc_cnn = 512
    text_enc_dropout = 0.5
    text_enc_weights_path = 'new_text_enc.pth'
    text_dec_weights_path = 'faster_text_gen_v1.pth'
    lat_gen_weights_path = 'run_12_all_fixed/epoch_46_lat_gen.pth'
    lat_dis_weights_path = 'run_12_all_fixed/epoch_46_lat_dis.pth'
    references_path = QUORA_TEXT_PRETRAINED_VOCAB_VALID_SET_PATH

    text_enc = TextEncoder(d_vocab=d_vocab, d_text_feature=d_text_feature, text_enc_dropout=text_enc_dropout, d_text_enc_cnn=d_text_enc_cnn).to(device)
    text_enc.load_state_dict(torch.load(text_enc_weights_path))
    text_dec = TextDecoder(d_vocab=d_vocab, d_text_feature=d_text_feature, d_gen_hidden=d_gen_hidden, d_max_seq_len=d_max_seq_len, d_gen_layers=d_gen_layers, gen_dropout=gen_dropout, pad_token=dataset.pad_token, start_token=dataset.start_token, end_token=dataset.end_token).to(device)
    text_dec.load_state_dict(torch.load(text_dec_weights_path))

    lat_dis = LatentDiscriminator(d_text_feature=d_text_feature, d_dis_hidden=d_dis_hidden).to(device)
    lat_dis.load_state_dict(torch.load(lat_dis_weights_path))
    lat_gen = LatentGenerator(d_noise=d_noise, d_text_feature=d_text_feature, d_gen_hidden=d_gen_hidden).to(device)
    lat_gen.load_state_dict(torch.load(lat_gen_weights_path))

    print('training:')
    train(lat_gen, text_dec, text_enc, lat_dis, dataset, loader, device, output_dir=output_dir, references=references, **config)
    print('finished training')


# # Runs

if __name__ == '__main__':
    run(config=dict(num_epochs=50), output_dir='run_13/')
# # End
