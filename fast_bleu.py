import numpy as np

from tqdm import tqdm
from collections import Counter

def best_match_len(refs, hypo):
    min_diff = 10000
    hlen = len(hypo)
    for ref in refs:
        rlen = len(ref)
        curr_diff = abs(hlen - rlen)
        if curr_diff < min_diff or (curr_diff == min_diff and rlen < r):
            min_diff = curr_diff
            r = rlen
    return r

def brevity(refs, hypo):
    r = best_match_len(refs, hypo)
    return min(1.0, np.exp(1 - (r/len(hypo))))

def get_ngrams(xs, n):
    end = len(xs) - n + 1
    if end <= 0:
        return []
    xs = tuple(xs)
    ngrams = [xs[i:i+n] for i in range(end)]
    return ngrams

def pre_calc_max_cnts(refs, n):
    max_cnt = Counter()
    for ref in refs:
        ref_cnt = Counter(get_ngrams(ref, n))
        for ngram, cnt in ref_cnt.items():
            max_cnt[ngram] = max(cnt, max_cnt[ngram])
    return max_cnt

def get_clipped_score_using_pre_calc(max_cnt, hypo, n):
    hypo_cnt = Counter(get_ngrams(hypo, n))
    return sum(min(cnt, max_cnt[ngram]) for ngram, cnt in hypo_cnt.items())

def mod_pre_using_pre_calc(max_cnt, hypo, n):
    score = get_clipped_score_using_pre_calc(max_cnt, hypo, n)
    total = len(hypo) - n + 1
    return score/total

def bleu_using_pre_calc(max_cnts, refs, hypo, n):
    if len(hypo) < n:
        return 0.0
    ps = [mod_pre_using_pre_calc(max_cnt, hypo, i) for i, max_cnt in enumerate(max_cnts, start=1)]
    bp = brevity(refs, hypo)
    return np.prod(ps)**(1.0/n) * bp

def bleu_with_common_refs(refs, hypos, N):
    max_cnts = [pre_calc_max_cnts(refs, n) for n in range(1, N+1)]
    return [bleu_using_pre_calc(max_cnts, refs, hypo, N) for hypo in tqdm(hypos)]

def all_bleu_using_pre_calc(max_cnts, refs, hypo):
    hlen = len(hypo)
    N = len(max_cnts)
    ps = [mod_pre_using_pre_calc(max_cnt, hypo, n) if hlen >= n else 0.0 for n, max_cnt in enumerate(max_cnts, start=1)]
    bp = brevity(refs, hypo) if len(hypo) > 0 else 0.0
    return [(np.prod(ps[:n])**(1.0/n) * bp) for n in range(1, N+1)]

def all_bleu_with_common_refs(refs, hypos, N):
    max_cnts = [pre_calc_max_cnts(refs, n) for n in range(1, N+1)]
    return [all_bleu_using_pre_calc(max_cnts, refs, hypo) for hypo in tqdm(hypos)]

# Faster version

# def faster_best_match_len(refs, hypo):
#     hlen = len(hypo)
#     def keyfunc(rlen):
#         return abs(rlen - hlen), rlen
#     return min([len(ref) for ref in refs], key=keyfunc)

# Could also calculate lengths and all brevity penalties together.
