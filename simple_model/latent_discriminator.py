import torch

from torch import nn
from utils import to_one_hot
from text_encoder import TextEncoder

class SimpleLatentDiscriminator(nn.Module):
    def __init__(self, d_text_feature, **kwargs):
        super().__init__()
        self.d_text_feature = d_text_feature
        self.define_module()

    def define_module(self):
        self.fc_logits = nn.Linear(self.d_text_feature, 1)

    def forward(self, xs):
        return self.fc_logits(xs).squeeze(1)

class LatentDiscriminator(nn.Module):
    def __init__(self, d_text_feature, d_dis_hidden, dis_dropout=0.5, **kwargs):
        super().__init__()
        self.d_text_feature = d_text_feature
        self.d_dis_hidden = d_dis_hidden
        self.dis_dropout = dis_dropout
        self.define_module()

    def define_module(self):
        slope = 0.2
        self.fc_logits = nn.Sequential(
            nn.Linear(self.d_text_feature, self.d_dis_hidden),
            nn.Dropout(self.dis_dropout),
            nn.LeakyReLU(slope),
            nn.Linear(self.d_dis_hidden, self.d_dis_hidden),
            nn.Dropout(self.dis_dropout),
            nn.LeakyReLU(slope),
            nn.Linear(self.d_dis_hidden, self.d_dis_hidden),
            nn.Dropout(self.dis_dropout),
            nn.LeakyReLU(slope),
            nn.Linear(self.d_dis_hidden, 1)
        )

    def forward(self, xs):
        return self.fc_logits(xs).squeeze(1)

def test_latent_discriminator():
    d_batch = 2
    d_max_seq_len = 26
    d_vocab = 27699
    d_dis_hidden = 256
    d_text_feature = 512
    text_enc_dropout = 0.5
    d_text_enc_cnn = 512

    lat_dis = LatentDiscriminator(d_text_feature=d_text_feature, d_dis_hidden=d_dis_hidden)

    text_features = torch.randn(d_batch, d_text_feature)
    valids = lat_dis(text_features)
    assert valids.size() == (d_batch,) and valids.dtype == torch.float

def test_text_dis_end_to_end():
    d_batch = 2
    d_max_seq_len = 26
    d_vocab = 27699
    d_dis_hidden = 256
    d_text_feature = 512
    text_enc_dropout = 0.5
    d_text_enc_cnn = 512

    text_enc = TextEncoder(d_vocab=d_vocab, d_text_feature=d_text_feature, text_enc_dropout=text_enc_dropout, d_text_enc_cnn=d_text_enc_cnn)
    lat_dis = LatentDiscriminator(d_text_feature=d_text_feature, d_dis_hidden=d_dis_hidden)
    text_enc.load_state_dict(torch.load('new_text_enc.pth'))

    texts = torch.randint(low=0, high=d_vocab, size=(d_batch, d_max_seq_len))
    text_features = text_enc(to_one_hot(texts, d_vocab))
    assert text_features.size() == (d_batch, d_text_feature) and text_features.dtype == torch.float

    valids = lat_dis(text_features)
    assert valids.size() == (d_batch,) and valids.dtype == torch.float
