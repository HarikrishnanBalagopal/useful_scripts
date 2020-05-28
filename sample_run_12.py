import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from tqdm.notebook import tqdm
from scipy.stats import truncnorm
from argparse import ArgumentParser
from text_encoder import TextEncoder
from text_decoder import TextDecoder
from easydict import EasyDict as edict
from fast_self_bleu import all_self_bleu
from fast_bleu import all_bleu_with_common_refs
from latent_discriminator import LatentDiscriminator, SimpleLatentDiscriminator
from latent_generator import LatentGenerator, SimpleLatentGenerator1, SimpleLatentGenerator2
from text_datasets import get_quora_texts_pretrained_vocab, get_emnlp_2017_news_pretrained_vocab
from utils import to_one_hot, QUORA_TEXT_PRETRAINED_VOCAB_VALID_SET_PATH, EMNLP_2017_NEWS_PRETRAINED_VOCAB_VALID_SET_PATH

def run(args):
    d_batch = 512
    d_noise = 100
    d_vocab = 27699
    d_gen_layers = 1
    gen_dropout = 0.5
    d_max_seq_len = 26
    d_gen_hidden = 512
    d_dis_hidden = 512
    d_text_feature = 512
    d_text_enc_cnn = 512
    d_text_enc_cnn = 512
    text_enc_dropout = 0.5
    output_dir = args.output_dir
    interpolation_steps = d_batch
    lat_gen_weights_path = args.weights_path
    text_enc_weights_path = 'new_text_enc.pth'
    text_dec_weights_path = 'faster_text_gen_v1.pth'
    truncation_threshold = args.truncation_threshold

    if truncation_threshold is not None:
        assert truncation_threshold > 0.0, f'truncation_threshold must be positive'
        truncated_normal = truncnorm(-truncation_threshold, truncation_threshold)
        print('using truncation trick with threshold:', truncation_threshold)

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset, loader = get_quora_texts_pretrained_vocab(split='train', d_batch=d_batch, should_pad=True, pad_to_length=d_max_seq_len)
    assert d_vocab == dataset.d_vocab
    end_token = dataset.end_token

    text_dec = TextDecoder(d_vocab=d_vocab, d_text_feature=d_text_feature, d_gen_hidden=d_gen_hidden, d_max_seq_len=d_max_seq_len, d_gen_layers=d_gen_layers, gen_dropout=gen_dropout, pad_token=dataset.pad_token, start_token=dataset.start_token, end_token=dataset.end_token).to(device)
    text_dec.load_state_dict(torch.load(text_dec_weights_path))

    lat_gen = LatentGenerator(d_noise=d_noise, d_text_feature=d_text_feature, d_gen_hidden=d_gen_hidden).to(device)
    lat_gen.load_state_dict(torch.load(lat_gen_weights_path))
#     lat_gen = SimpleLatentGenerator1(d_noise=d_noise, d_text_feature=d_text_feature, d_gen_hidden=d_gen_hidden).to(device)
#     lat_gen = SimpleLatentGenerator2(d_noise=d_noise, d_text_feature=d_text_feature, d_gen_hidden=d_gen_hidden).to(device)

    text_dec.eval()
    lat_gen.eval()

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

    def get_avg_std_bleu(xss):
        xss = list(zip(*xss))
        return [(np.mean(xs), np.std(xs)) for xs in xss]

    print('generate fake texts:')
    fake_text_ints = []
    fake_text_strs = []
    num_generated = 0
    num_samples = args.num_samples
    with torch.no_grad():
        while num_generated < num_samples:
            if truncation_threshold is not None:
                noises = torch.tensor(truncated_normal.rvs((d_batch, d_noise)), dtype=torch.float, device=device)
            else:
                noises = torch.randn(d_batch, d_noise, device=device)
            fake_texts = text_dec(lat_gen(noises))[0]
            fake_text_ints.extend(fake_texts.tolist())
            fake_text_strs.extend([dataset.decode_caption(fake_text) for fake_text in fake_texts.cpu()])
            num_generated += d_batch
    print('\n'.join(fake_text_strs[:4]))
    with open(f'{output_dir}/fake_texts.txt', 'w') as f:
        f.write('\n'.join(fake_text_strs))
    fake_text_ints = remove_special_tokens(fake_text_ints)
    torch.save(fake_text_ints, f'{output_dir}/fake_texts.pth')
    print('DONE!')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-w', dest='weights_path', type=str, required=True, help='path to weights')
    parser.add_argument('-o', dest='output_dir', type=str, required=True, help='output folder name')
    parser.add_argument('--truncation_threshold', type=float, default=None, help='threshold for the truncation trick, default is no truncation.')
    parser.add_argument('-n', dest='num_samples', type=int, default=5000, help='number of samples to generate')
    return parser.parse_args()

if __name__ == '__main__':
    run(parse_args())
