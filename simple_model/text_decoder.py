import torch

from torch import nn
from utils import to_one_hot
from text_datasets import get_quora_texts_pretrained_vocab

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

def test_mask_in_place_and_calc_length():
    d_batch = 2
    d_max_seq_len = 8
    d_vocab = 4
    pad_token = 0
    eos_token = 2

    texts = torch.randint(low=0, high=d_vocab, size=(d_batch, d_max_seq_len))
    text_log_probs = F.log_softmax(torch.randn(d_batch, d_max_seq_len, d_vocab), dim=2)
    print(texts)
    print(text_log_probs)

    texts, text_log_probs, text_lens = mask_in_place_and_calc_length(texts, text_log_probs, eos_token, pad_token)
    assert text_lens.size() == (d_batch,) and text_lens.dtype == torch.long

    print('text_lens:', text_lens)
    print(texts)
    print(text_log_probs)

class TextDecoder(nn.Module):
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
        texts = torch.full((d_batch, d_max_seq_len), self.pad_token, dtype=torch.long, device=device)
        hs = None

        for i in range(0, d_max_seq_len):
            if i > 0:
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

def test_text_decoder():
    d_batch = 4
    d_max_seq_len = 52
    d_text_feature = 512
    d_gen_hidden = 512
    gen_dropout = 0.5
    d_gen_layers = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset, _ = get_quora_texts_pretrained_vocab(d_batch=d_batch, should_pad=True, pad_to_length=d_max_seq_len)

    text_dec = TextDecoder(d_vocab=dataset.d_vocab, d_text_feature=d_text_feature, d_gen_hidden=d_gen_hidden, d_max_seq_len=dataset.d_max_seq_len, d_gen_layers=d_gen_layers, gen_dropout=gen_dropout, pad_token=dataset.pad_token, start_token=dataset.start_token, end_token=dataset.end_token)
    text_dec.load_state_dict(torch.load('faster_text_gen_v1.pth'))
    text_dec.to(device).train()

    text_features = torch.randn(d_batch, d_text_feature, device=device)

    texts, text_log_probs, text_lens = text_dec(text_features)
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
