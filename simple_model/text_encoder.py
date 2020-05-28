import torch

from torch import nn
from utils import to_one_hot

class TextEncoder(nn.Module):
    def __init__(self, d_vocab, d_text_feature, text_enc_dropout=0.0, d_text_enc_cnn=512):
        super().__init__()
        self.d_vocab = d_vocab
        self.d_text_enc_cnn = d_text_enc_cnn
        self.d_text_feature = d_text_feature
        self.text_enc_dropout = text_enc_dropout
        self.define_module()

    def define_module(self):
        self.conv1 = nn.Conv1d(self.d_vocab, 256, 1)
        self.threshold = nn.Threshold(0.000001, 0)
        self.conv2 = nn.Conv1d(256, self.d_text_enc_cnn, 1)
        self.rnn = nn.GRU(self.d_text_enc_cnn, self.d_text_enc_cnn)
        self.linear = nn.Linear(self.d_text_enc_cnn, self.d_text_feature)
        self.dropout_layer = nn.Dropout(self.text_enc_dropout)

    def forward(self, xs):
        xs = self.threshold(self.conv2(self.threshold(self.conv1(xs.transpose(1, 2))))).permute(2, 0, 1)
        xs = self.rnn(xs)[0].mean(dim=0)
        xs = self.linear(self.dropout_layer(xs))
        return xs

def test_text_encoder():
    d_batch = 2
    d_max_seq_len = 26
    d_vocab = 27699
    d_text_feature = 512
    text_enc_dropout = 0.5
    d_text_enc_cnn = 512
    text_enc = TextEncoder(d_vocab=d_vocab, d_text_feature=d_text_feature, text_enc_dropout=text_enc_dropout, d_text_enc_cnn=d_text_enc_cnn)
    text_enc.load_state_dict(torch.load('new_text_enc.pth'))

    texts = torch.randint(low=0, high=d_vocab, size=(d_batch, d_max_seq_len))
    text_features = text_enc(to_one_hot(texts, d_vocab))
    assert text_features.size() == (d_batch, d_text_feature) and text_features.dtype == torch.float
