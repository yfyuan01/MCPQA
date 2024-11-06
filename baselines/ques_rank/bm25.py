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
def get_implict_matrix(rec_items, test_set):
    rel_matrix = [[0] * rec_items.shape[1] for _ in range(rec_items.shape[0])]
    for user in range(len(test_set)):
        for index, item in enumerate(rec_items[user]):
            if item in test_set[user]:
                rel_matrix[user][index] = 1
    return np.array(rel_matrix)


def DCG(items):
    return np.sum(items / np.log(np.arange(2, len(items) + 2)))


def nDCG(rec_items, test_set):
    assert rec_items.shape[0] == len(test_set)
    rel_matrix = get_implict_matrix(rec_items, test_set)
    ndcgs = []
    for user in range(len(test_set)):
        rels = rel_matrix[user]
        dcg = DCG(rels)
        idcg = DCG(sorted(rels, reverse=True))
        ndcg = dcg / idcg if idcg != 0 else 0
        ndcgs.append(ndcg)
    return ndcgs

def ndcg(golden, current, n = -1):
    log2_table = np.log2(np.arange(2, 102))
    def dcg_at_n(rel, n):
        rel = np.asfarray(rel)[:n]
        dcg = np.sum(np.divide(np.power(2, rel) - 1, log2_table[:rel.shape[0]]))
        return dcg
    ndcgs = []
    for i in range(len(current)):
        k = len(current[i]) if n == -1 else n
        idcg = dcg_at_n(sorted(golden[i], reverse=True), n=k)
        dcg = dcg_at_n(current[i], n=k)
        tmp_ndcg = 0 if idcg == 0 else dcg / idcg
        ndcgs.append(tmp_ndcg)
    return 0. if len(ndcgs) == 0 else sum(ndcgs) / (len(ndcgs))
def change_score(i):
    if i == 0:
        return 0
    elif i == 1:
        return 0.5
    elif i== 2:
        return 1
for c in country1:
    print(c)
    mrr_all = 0
    p_all = 0
    ques_true = {}
    ques_true_idx = {}
    ques_pre = {}
    ques_true_pos_idx = {}
    test_file = os.path.join(test_file_address,'results_'+c+'.jsonl')
    pre_file = os.path.join(test_file_address+'/intermediate',c+'_quesrank_cross_market.jsonl')
    test_relevance = [json.loads(i) for i in open(test_file).readlines()]
    pre_relevance = [json.loads(i) for i in open(pre_file).readlines()]
    for k in tqdm((range(0,len(test_relevance),5))):
        t = test_relevance[int(k/5)]
        ques_true[int(k/5)] = [i['gpt4_score'] for i in test_relevance[k:k+5]]
        if max([i['gpt4_score'] for i in test_relevance[k:k+5]])==1:
            idx = [j for j in range(len(ques_true[int(k / 5)])) if ques_true[int(k / 5)][j] == 1]
        else:
            idx = [j for j in range(len(ques_true[int(k/5)])) if ques_true[int(k/5)][j]==2]
        ques_true_idx[int(k/5)] = idx
        score_array = np.array([i['gpt4_score'] for i in test_relevance[k:k + 5]])
        eq_letter1 = list(np.where(score_array == 2)[0])
        eq_letter2 = list(np.where(score_array == 1)[0])
        eq_letter3 = list(np.where(score_array == 0)[0])
        ques_true_pos_idx[int(k / 5)] = eq_letter1 + eq_letter2
        # print(ques_true[k])
    # print(len(ques_true))
    for k,t in tqdm(enumerate(pre_relevance)):
        # tokenized_query = t['question'].split(' ')
        # tokenized_corpus = [doc.split(" ") for doc in t['bm25_top5']]
        # bm25 = BM25Okapi(tokenized_corpus)
        # doc_scores = bm25.get_scores(tokenized_query)
        if ques_true_idx[k]==[]:
            p3 = 0
            mrr = 0
        else:
            mrr = 1.0 / (min([j for j in ques_true_idx[k]]) + 1)
            p = len([id for id in range(3) if id in ques_true_pos_idx[k]]) / 3.0
    # for k,t in tqdm(enumerate(pre_relevance)):
    #     tokenized_query = t['translatedQuestion'].split(' ')
    #     tokenized_corpus = [doc.split(" ") for doc in t['bm25_top5']]
    #     bm25 = BM25Okapi(tokenized_corpus)
    #     doc_scores = bm25.get_scores(tokenized_query)
    #     doc_scores_idx = {doc_scores[j]:j for j in range(len(doc_scores))}
    #     ranked_doc_scores_idx = [doc_scores_idx[s] for s in sorted(doc_scores,reverse=True)]
    #     if ques_true_idx[k] == []:
    #         p3 = 0
    #         mrr = 0
    #     else:
    #         mrr_inv = max([doc_scores[j] for j in ques_true_idx[k]])
    #         p = len([id for id in ranked_doc_scores_idx[:3] if id in ques_true_pos_idx[k]])/3.0
    #         mrr_inv = sorted(doc_scores, reverse=True).index(mrr_inv) + 1
    #         mrr = 1 / float(mrr_inv)
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
    # [0,1,1,0,0.5]->[1,2,4,0,3]
    ques_true_pos_idx = {}
    ques_pre = {}
    test_file = os.path.join(test_file_address,'results_'+c+'.jsonl')
    pre_file = os.path.join(test_file_address+'/intermediate',c+'_quesrank_cross_market.jsonl')
    test_relevance = [json.loads(i) for i in open(test_file).readlines()]
    pre_relevance = [json.loads(i) for i in open(pre_file).readlines()]
    for k in tqdm((range(0,len(test_relevance),5))):
        t = test_relevance[int(k/5)]
        ques_true[int(k/5)] = [i['gpt4_score'] for i in test_relevance[k:k+5]]
        if max([i['gpt4_score'] for i in test_relevance[k:k+5]])==1:
            idx = [j for j in range(len(ques_true[int(k / 5)])) if ques_true[int(k / 5)][j] == 1]
        else:
            idx = [j for j in range(len(ques_true[int(k/5)])) if ques_true[int(k/5)][j]==2]
        ques_true_idx[int(k/5)] = idx
        score_array = np.array([i['gpt4_score'] for i in test_relevance[k:k + 5]])
        eq_letter1 = list(np.where(score_array==2)[0])
        eq_letter2 = list(np.where(score_array==1)[0])
        eq_letter3 = list(np.where(score_array == 0)[0])
        # ques_true_idx_dict = Counter([i['gpt4_score'] for i in test_relevance[k:k+5]])
        # ques_true_pos = [[1]*ques_true_idx_dict[1]]+[[0.5]*ques_true_idx_dict[0.5]]+[[0]*ques_true_idx_dict[0]]
        ques_true_pos_idx[int(k/5)] = eq_letter1+eq_letter2
        # print(ques_true[k])
    # print(len(ques_true))
    for k,t in tqdm(enumerate(pre_relevance)):
        # tokenized_query = t['question'].split(' ')
        # tokenized_corpus = [doc.split(" ") for doc in t['bm25_top5']]
        # bm25 = BM25Okapi(tokenized_corpus)
        # doc_scores = bm25.get_scores(tokenized_query)
        if ques_true_idx[k]==[]:
            p3 = 0
            mrr = 0
        else:
            mrr = 1.0 / (min([j for j in ques_true_idx[k]]) + 1)
            p = len([id for id in range(3) if id in ques_true_pos_idx[k]]) / 3.0
        # doc_scores_idx = {doc_scores[j]:j for j in range(len(doc_scores))}
        # ranked_doc_scores_idx = [doc_scores_idx[s] for s in sorted(doc_scores,reverse=True)]
        # print(ranked_doc_scores_idx)
        # print(doc_scores)
        # print(doc_scores)
        # print(ques_true[k])
        # print(ques_true_pos_idx[k])
        # print(ques_true_idx[k])
        # [0,0.5,1,0.5,1] -> [2,4]  -> index 2,4 in original ranking  [3,2,1,4,0]
        # if ques_true_idx[k]==[]:
        #     p3 = 0
        #     mrr = 0
        # else:
        #     mrr_inv = max([doc_scores[j] for j in ques_true_idx[k]])
        #     p = len([id for id in ranked_doc_scores_idx[:3] if id in ques_true_pos_idx[k]]) / 3.0
        #     mrr_inv = sorted(doc_scores,reverse=True).index(mrr_inv)+1
        #     # mrr_inv = sorted(doc_scores,reverse=True).index(mrr_inv[0])+1
        #     mrr = 1/float(mrr_inv)
        #     mrr1 = 1.0/(min([j for j in ques_true_idx[k]])+1)
        #     print(ques_true[k])
        #     print(mrr)
        #     print(mrr1)
        #     print('---------')
        #     break
            # print(mrr)
        # break
        mrr_all+=mrr
        p_all+=p
    mrr_all/=len(pre_relevance)
    p_all/=len(pre_relevance)
    print(mrr_all)
    print(p_all)
    print('*************')
        # break

