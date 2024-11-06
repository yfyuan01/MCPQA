from rank_bm25 import BM25Okapi
import numpy as np
import json
# from sklearn.metrics import ndcg_score
import os
from tqdm import tqdm
import math
from collections import Counter
class Model:
    def __init__(self, k):
        self.k = k
        self.item_size = 50

    def __call__(self, users):
        res = np.random.randint(0, self.item_size, users.shape[0] * self.k)
        return res.reshape((users.shape[0], -1))

test_file_address = '../../LLMdata/ques200/'
country = ['au','ca','uk','in']
country1 = ['br','cn','fr','jp','mx']
market = 'cross'
for c in country1:
    print(c)
    mrr_all = 0
    p_all = 0
    ques_true = {}
    ques_sent = {}
    ques_true_idx = {}
    ques_sent_pos = {}
    ques_pre = {}
    ques_true_pos_idx = {}
    ques_sent_pos_true = {}
    test_file = os.path.join(test_file_address,'results_'+c+'.jsonl')
    pre_file = os.path.join(test_file_address+'/intermediate',c+'_quesrank_'+market+'_market_new.jsonl')
    test_relevance = [json.loads(i) for i in open(test_file).readlines()]
    pre_relevance = [json.loads(i) for i in open(pre_file).readlines()]
    for k in tqdm((range(0,len(test_relevance),5))):
        t = test_relevance[int(k/5)]
        ques_true[int(k/5)] = [i['gpt4_score'] for i in test_relevance[k:k+5]]
        ques_sent[int(k / 5)] = [i['relatedQuestion'] for i in test_relevance[k:k + 5]]
        if max([i['gpt4_score'] for i in test_relevance[k:k+5]])==1:
            idx = [j for j in range(len(ques_true[int(k / 5)])) if ques_true[int(k / 5)][j] == 1]
        else:
            idx = [j for j in range(len(ques_true[int(k/5)])) if ques_true[int(k/5)][j]==2]
        ques = [ques_sent[int(k/5)][j] for j in idx]
        ques_true_idx[int(k/5)] = idx
        ques_sent_pos[int(k/5)] = ques
        score_array = np.array([i['gpt4_score'] for i in test_relevance[k:k + 5]])
        eq_letter1 = list(np.where(score_array == 2)[0])
        eq_letter2 = list(np.where(score_array == 1)[0])
        eq_letter3 = list(np.where(score_array == 0)[0])
        ques_true_pos_idx[int(k / 5)] = eq_letter1 + eq_letter2
        ques_sent_pos_true[int(k / 5)] = [ques_sent[int(k / 5)][j] for j in eq_letter1 + eq_letter2]
        # print(ques_true_pos_idx[int(k / 5)])
        # print(ques_sent_pos_true[int(k / 5)])
        # print(ques_true[k])
    # print(len(ques_true))
    for k,t in tqdm(enumerate(pre_relevance)):
        p3 = 0
        mrr = 0
        if market == 'single':
            pre = [pp.split(' <SEP> ')[0] for pp in t['translated_bm25_top5']]
        else:
            pre = [pp.split(' <SEP> ')[0] for pp in t['bm25_top5']]
        for m, n in enumerate(ques_true_idx[k]):
            if pre[n] == ques_sent_pos[k][m]:
                mrr = 1.0 / (n + 1)
                break
        # for m,n in enumerate(ques_true_pos_idx[k]):
        pre = pre[:3]
        p = len([pp for pp in pre if pp in ques_sent_pos_true[k]]) / 3.0
        mrr_all += mrr
        p_all += p
    mrr_all/=len(pre_relevance)
    p_all /= len(pre_relevance)
    print(mrr_all)
    print(p_all)
    print('*****')
        # break

for c in country:
    print(c)
    mrr_all = 0
    p_all = 0
    ques_true = {}
    ques_true_idx = {}
    ques_true_pos_idx = {}
    ques_pre = {}
    ques_sent = {}
    ques_sent_pos = {}
    ques_sent_pos_true = {}
    test_file = os.path.join(test_file_address,'results_'+c+'.jsonl')
    pre_file = os.path.join(test_file_address+'/intermediate',c+'_quesrank_'+market+'_market_new.jsonl')
    test_relevance = [json.loads(i) for i in open(test_file).readlines()]
    pre_relevance = [json.loads(i) for i in open(pre_file).readlines()]
    for k in tqdm((range(0,len(test_relevance),5))):
        t = test_relevance[int(k/5)]
        ques_true[int(k/5)] = [i['gpt4_score'] for i in test_relevance[k:k+5]]
        ques_sent[int(k/5)] = [i['relatedQuestion'] for i in test_relevance[k:k+5]]
        if max([i['gpt4_score'] for i in test_relevance[k:k+5]])==1:
            idx = [j for j in range(len(ques_true[int(k / 5)])) if ques_true[int(k / 5)][j] == 1]
        else:
            idx = [j for j in range(len(ques_true[int(k/5)])) if ques_true[int(k/5)][j]==2]
        ques = [ques_sent[int(k / 5)][j] for j in idx]
        ques_true_idx[int(k/5)] = idx
        ques_sent_pos[int(k/5)] = ques
        score_array = np.array([i['gpt4_score'] for i in test_relevance[k:k + 5]])
        eq_letter1 = list(np.where(score_array==2)[0])
        eq_letter2 = list(np.where(score_array==1)[0])
        eq_letter3 = list(np.where(score_array == 0)[0])
        # ques_true_idx_dict = Counter([i['gpt4_score'] for i in test_relevance[k:k+5]])
        # ques_true_pos = [[1]*ques_true_idx_dict[1]]+[[0.5]*ques_true_idx_dict[0.5]]+[[0]*ques_true_idx_dict[0]]
        ques_true_pos_idx[int(k/5)] = eq_letter1+eq_letter2
        ques_sent_pos_true[int(k/5)] = [ques_sent[int(k/5)][j] for j in eq_letter1+eq_letter2]
        # print(ques_true[k])
    # print(len(ques_true))
    for k,t in tqdm(enumerate(pre_relevance)):
        p3 = 0
        mrr = 0
        pre = [pp.split(' <SEP> ')[0] for pp in t['bm25_top5']]
        for m,n in enumerate(ques_true_idx[k]):
            if pre[n] == ques_sent_pos[k][m]:
                mrr = 1.0/(n+1)
                break
        # for m,n in enumerate(ques_true_pos_idx[k]):
        pre = pre[:3]
        p = len([pp for pp in pre if pp in ques_sent_pos_true[k]])/3.0
        mrr_all+=mrr
        p_all+=p
    mrr_all/=len(pre_relevance)
    p_all/=len(pre_relevance)
    print(mrr_all)
    print(p_all)
    print('*************')
        # break

