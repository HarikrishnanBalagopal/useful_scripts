import numpy as np

from tqdm import tqdm
from collections import Counter, defaultdict

def get_ngrams(xs, n):
    end = len(xs) - n + 1
    if end <= 0:
        return []
    xs = tuple(xs)
    ngrams = [xs[i:i+n] for i in range(end)]
    return ngrams

def pre_calc_max_cnts(refs, n):
    max_cnt = {}
    for i, ref in enumerate(refs):
        ref_cnt = Counter(get_ngrams(ref, n))
        for ngram, cnt in ref_cnt.items():
            if ngram not in max_cnt:
                max_cnt[ngram] = (i, cnt, -1, 0)
            else:
                i1, c1, i2, c2 = max_cnt[ngram]
                if cnt > c1:
                    max_cnt[ngram] = (i, cnt, i1, c1)
                elif cnt > c2:
                    max_cnt[ngram] = (i1, c1, i, cnt)
    return max_cnt

def exclude_ref(max_cnt, i):
    res = Counter()
    for ngram, (i1, c1, i2, c2) in max_cnt.items():
        res[ngram] = c1 if i != i1 else c2
    return res

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

def get_clipped_score_using_pre_calc(max_cnt, hypo, n):
    hypo_cnt = Counter(get_ngrams(hypo, n))
    return sum(min(cnt, max_cnt[ngram]) for ngram, cnt in hypo_cnt.items())

def mod_pre_using_pre_calc(max_cnt, hypo, n):
    score = get_clipped_score_using_pre_calc(max_cnt, hypo, n)
    total = len(hypo) - n + 1
    return score/total

def self_bleu(hypos, N):
    max_cnts = [pre_calc_max_cnts(hypos, n) for n in range(1, N+1)]
    scores = []
    for i, hypo in enumerate(tqdm(hypos)):
        if len(hypo) < N:
            scores.append(0.0)
        else:
            ps = [mod_pre_using_pre_calc(exclude_ref(max_cnt, i), hypo, n) for n, max_cnt in enumerate(max_cnts, start=1)]
            bp = brevity(hypos[:i] + hypos[i+1:], hypo) if len(hypo) > 0 else 0.0
            scores.append(np.prod(ps)**(1.0/N) * bp)
    return scores

def all_self_bleu(hypos, N):
    max_cnts = [pre_calc_max_cnts(hypos, n) for n in range(1, N+1)]
    scores = []
    for i, hypo in enumerate(tqdm(hypos)):
        hlen = len(hypo)
        ps = [mod_pre_using_pre_calc(exclude_ref(max_cnt, i), hypo, n) if hlen >= n else 0.0 for n, max_cnt in enumerate(max_cnts, start=1)]
        bp = brevity(hypos[:i] + hypos[i+1:], hypo) if len(hypo) > 0 else 0.0
        scores.append([(np.prod(ps[:n])**(1.0/n) * bp) for n in range(1, N+1)])
    return scores
