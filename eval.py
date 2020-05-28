import os
import json
import torch
import numpy as np

from tqdm import tqdm
from argparse import ArgumentParser
from fast_self_bleu import self_bleu, all_self_bleu
from fast_bleu import bleu_with_common_refs, all_bleu_with_common_refs

def eval_single_score(refs, hypos, n):
    scores = bleu_with_common_refs(refs, hypos, n)
    scores_mean = np.mean(scores)
    scores_std = np.std(scores)
    self_scores = self_bleu(hypos, N=n)
    self_scores_mean = np.mean(self_scores)
    self_scores_std = np.std(self_scores)
    print(f'BLEU {n} score:', scores_mean, 'std:', scores_std)
    print(f'Self BLEU {i} score:', self_scores_mean, 'std:', self_scores_std)
    bleu_scores= dict(scores=scores, scores_mean=scores_mean, scores_std=scores_std)
    bleu_scores= dict(scores=scores, scores_mean=scores_mean, scores_std=scores_std, self_scores=self_scores, self_scores_mean=self_scores_mean, self_scores_std=self_scores_std)
    return dict(n=n, bleu_scores=bleu_scores)

def eval_scores_upto_n(refs, hypos, n):
    bleu_scores = {}
    for i in range(1, n+1):
        scores = bleu_with_common_refs(refs, hypos, i)
        scores_mean = np.mean(scores)
        scores_std = np.std(scores)
        self_scores = self_bleu(hypos, N=i)
        self_scores_mean = np.mean(self_scores)
        self_scores_std = np.std(self_scores)
        bleu_scores[i] = dict(scores=scores, scores_mean=scores_mean, scores_std=scores_std)
        bleu_scores[i] = dict(scores=scores, scores_mean=scores_mean, scores_std=scores_std, self_scores=self_scores, self_scores_mean=self_scores_mean, self_scores_std=self_scores_std)
        print(f'BLEU {i} score:', scores_mean, 'std:', scores_std)
        print(f'Self BLEU {i} score:', self_scores_mean, 'std:', self_scores_std)
    return dict(n=n, bleu_scores=bleu_scores)

def load_data(references_path, hypotheses_path, limit):
    refs = torch.load(references_path)[:limit]
    hypos = torch.load(hypotheses_path)[:limit]
    if limit > 0:
        refs = refs[:limit]
        hypos = hypos[:limit]
    return refs, hypos

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--refs', type=str, required=True, dest='references_path', help='path to references')
    parser.add_argument('--hypos', type=str, required=True, dest='hypotheses_path', help='path to hypotheses')
    parser.add_argument('-n', type=int, default=5, help='which BLEU to calculate (1, 2, 3, 4 or 5)')
    parser.add_argument('--limit', type=int, default=5000, help='max number of references and hypotheses to use. set this to -1 to use everything.')
    parser.add_argument('-o', type=str, dest='output_path', default='bleu_scores.json', help='output filename')
    return parser.parse_args()

def run():
    args = parse_args()
    assert os.path.isfile(args.references_path)
    assert os.path.isfile(args.hypotheses_path)
    assert args.n > 0 and args.n < 6 # this check can be removed, it is not necessary to restrict to less than 6

    print('running with config:', args)
    refs, hypos = load_data(references_path=args.references_path, hypotheses_path=args.hypotheses_path, limit=args.limit)

    results = dict(n=args.n, limit=args.limit)

    bleu_scores = all_bleu_with_common_refs(refs=refs, hypos=hypos, N=args.n)
    bleu_scores = zip(*bleu_scores)
    bleu_scores = {n:dict(bleu_score=bleu_score, mean=np.mean(bleu_score), std=np.std(bleu_score)) for n, bleu_score in enumerate(bleu_scores, start=1)}
    results['bleu_scores'] = bleu_scores
    for n in range(1, args.n+1):
        print(f'BLEU {n} score:', bleu_scores[n]['mean'], 'std:', bleu_scores[n]['std'])

    self_bleu_scores = all_self_bleu(hypos=hypos, N=args.n)
    self_bleu_scores = zip(*self_bleu_scores)
    self_bleu_scores = {n:dict(self_bleu_score=self_bleu_score, mean=np.mean(self_bleu_score), std=np.std(self_bleu_score)) for n, self_bleu_score in enumerate(self_bleu_scores, start=1)}
    results['self_bleu_scores'] = self_bleu_scores
    for n in range(1, args.n+1):
        print(f'Self BLEU {n} score:', self_bleu_scores[n]['mean'], 'std:', self_bleu_scores[n]['std'])

    with open(args.output_path, 'w') as f:
        json.dump(results, f)
    print('DONE!')

if __name__ == '__main__':
    run()
