import torch

from torch import nn
from utils import to_one_hot
from text_encoder import TextEncoder

class SimpleLatentGenerator1(nn.Module):
    def __init__(self, d_noise, d_text_feature, d_gen_hidden, gen_dropout=0.5, **kwargs):
        super().__init__()
        self.d_noise = d_noise
        self.d_text_feature = d_text_feature
        self.d_gen_hidden = d_gen_hidden
        self.gen_dropout = gen_dropout
        self.define_module()

    def define_module(self):
        slope = 0.2
        self.fc_text_features = nn.Linear(self.d_noise, self.d_text_feature)

    def forward(self, xs):
        return self.fc_text_features(xs)

class SimpleLatentGenerator2(nn.Module):
    def __init__(self, d_noise, d_text_feature, d_gen_hidden, gen_dropout=0.5, **kwargs):
        super().__init__()
        self.d_noise = d_noise
        self.d_text_feature = d_text_feature
        self.d_gen_hidden = d_gen_hidden
        self.gen_dropout = gen_dropout
        self.define_module()

    def define_module(self):
        slope = 0.2
        self.fc_text_features = nn.Sequential(
            nn.Linear(self.d_noise, self.d_gen_hidden),
            nn.Dropout(self.gen_dropout),
            nn.LeakyReLU(slope),
            nn.Linear(self.d_gen_hidden, self.d_text_feature),
        )

    def forward(self, xs):
        return self.fc_text_features(xs)

class LatentGenerator(nn.Module):
    def __init__(self, d_noise, d_text_feature, d_gen_hidden, gen_dropout=0.5, **kwargs):
        super().__init__()
        self.d_noise = d_noise
        self.d_text_feature = d_text_feature
        self.d_gen_hidden = d_gen_hidden
        self.gen_dropout = gen_dropout
        self.define_module()

    def define_module(self):
        slope = 0.2
        self.fc_text_features = nn.Sequential(
            nn.Linear(self.d_noise, self.d_gen_hidden),
            nn.Dropout(self.gen_dropout),
            nn.LeakyReLU(slope),
            nn.Linear(self.d_gen_hidden, self.d_gen_hidden),
            nn.Dropout(self.gen_dropout),
            nn.LeakyReLU(slope),
            nn.Linear(self.d_gen_hidden, self.d_gen_hidden),
            nn.Dropout(self.gen_dropout),
            nn.LeakyReLU(slope),
            nn.Linear(self.d_gen_hidden, self.d_text_feature)
        )

    def forward(self, xs):
        return self.fc_text_features(xs)

def test_latent_generator():
    d_batch = 2
    d_max_seq_len = 26
    d_vocab = 27699
    d_noise = 100
    d_gen_hidden = 256
    d_text_feature = 512
    text_enc_dropout = 0.5
    d_text_enc_cnn = 512

    lat_gen = LatentGenerator(d_noise=d_noise, d_text_feature=d_text_feature, d_gen_hidden=d_gen_hidden)

    noises = torch.randn(d_batch, d_noise)
    text_features = lat_gen(noises)
    assert text_features.size() == (d_batch, d_text_feature) and text_features.dtype == torch.float
