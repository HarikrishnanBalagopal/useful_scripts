import torch

from infersent import InferSent
from utils import INFERSENT_ENCODER_V2_PATH, INFERSENT_FAST_TEXT_W2V_PATH

def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.mm(torch.mm(u, torch.diag(si)), v.t())

def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.mm(sqrt_sigma, torch.mm(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

def frechet_distance(text_mean, text_cov, ref_mean, ref_cov):
    return torch.norm(text_mean - ref_mean) + torch.trace(text_cov) + torch.trace(ref_cov) - 2*trace_sqrt_product(text_cov, ref_cov)

def get_mean_and_cov(samples)
    mean = samples.mean(dim=0)
    errors = samples - mean.unsqueeze(0)
    cov = torch.mm(errors.t(), errors) / errors.size(0)
    return mean, cov

def get_infersent_features(text_strs, infersent):
    """
    infersent: the text encoder model
    text_strs: list(str)
    returns torch.Tensor of shape (d_batch, 4096) and dtype float
    """
    return torch.from_numpy(infersent.encode(text_strs, tokenize=True)).to(device)

def get_infersent():
    infersent = InferSent({'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, 'pool_type': 'max', 'dpout_model': 0.0, 'version': 2})
    infersent.load_state_dict(torch.load(INFERSENT_ENCODER_V2_PATH))
    infersent.set_w2v_path(INFERSENT_FAST_TEXT_W2V_PATH)
    infersent.build_vocab_k_words(K=100000)
    return infersent

def get_fed_score(text_strs, infersent, ref_mean, ref_cov):
    text_features = get_infersent_features(infersent, text_strs)
    text_mean, text_cov = get_mean_and_cov(text_features)
    return frechet_distance(text_mean, text_cov, ref_mean, ref_cov)
