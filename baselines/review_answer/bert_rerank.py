import json
from sentence_transformers import CrossEncoder
from scipy.special import softmax
import numpy as np
from tqdm import tqdm
import torch
import evaluate
from sacrebleu.tokenizers import tokenizer_zh,tokenizer_ja_mecab
from rouge_chinese import Rouge
import jieba
rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")
device = "cuda" if torch.cuda.is_available() else "cpu"
country1 = []
# country1 = ['au','ca','in','uk'] #'
country2 = ['cn','jp']
# country2 = ['br','fr','mx','cn','jp']
# country3 = ['cn','jp']
# model = CrossEncoder('nboost/pt-bert-base-uncased-msmarco',max_length=512)
model = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=512)
hypothesis = []
reference = []
for c in country1:
    print(c)
    # filename = '/home/dpl944/ProductQA/LLMdata/review1000/results_'+c+'.jsonl'
    # filename = '/home/dpl944/ProductQA/LLMdata/review1000/train_split/' + c + '_gpt4_bm25_review_item_aware_single_market_test.jsonl'
    filename = '/home/dpl944/ProductQA/Bert_classification/train_split/' + c + '_bm25_review_item_aware_single_market_test.jsonl'
    contents = [json.loads(i) for i in open(filename)]
    contents = [i for i in contents if i['topAnswer']!='']
    reference = [i['topAnswer'] for i in contents] #mark gpt4_answer
    hypothesis = []
    for i in tqdm(contents):
        pairs = []
        for k in i['bm25_top5']:
            pairs.append((i['question']+' '+i['topAnswer'],k)) #topAnswer
        scores = model.predict(pairs)
        scores = softmax(scores,axis=1)
        scores = np.argmax(scores,axis=0)
        hypo = i['bm25_top5'][scores[0]]
        hypothesis.append(hypo)
    r_results = rouge.compute(predictions=hypothesis, references=reference)
    b_results = bleu.compute(predictions=hypothesis, references=[[r] for r in reference], max_order=1, smooth=True)
    b = b_results["bleu"]
    r = r_results['rougeL']
    print(b)
    print(r)
    print('--------')
        # break

import pickle
import os
for c in country2:
    print(c)
#     # filename = '/home/dpl944/ProductQA/LLMdata/review1000/results_'+c+'.jsonl'
    filename = '/home/dpl944/ProductQA/Bert_classification/train_split/' + c + '_bm25_review_item_aware_single_market_test.jsonl'
    # filename = '/home/dpl944/ProductQA/LLMdata/review1000/train_split/' + c + '_gpt4_bm25_review_item_aware_single_market_test.jsonl'
    # t_dict = pickle.load(open(os.path.join('/home/dpl944/ProductQA/LLMdata/review1000/train_split/', 'translated_gpt_answer_' + c + '.pkl'), 'rb'))
    contents = [json.loads(i) for i in open(filename)]
    contents = [i for i in contents if i['topAnswer']!='']
    reference = [i['topAnswer'] for i in contents]
    # reference = [t_dict[i['topAnswer']] for i in contents] #mark'translatedAnswer'
    hypothesis = []
    for i in tqdm(contents):
        pairs = []
        for k in i['bm25_top5']: #mark translated_
            pairs.append((i['question']+' '+i['topAnswer'],k)) # mark translated
        scores = model.predict(pairs)
        scores = softmax(scores,axis=1)
        scores = np.argmax(scores,axis=0)
        hypo = i['bm25_top5'][scores[0]] #translated_ mark
        hypothesis.append(hypo)
    if c == 'cn':
        cn_rouge = Rouge()
        hypothesis = [' '.join(jieba.cut(h)) for h in hypothesis]
        reference = [' '.join(jieba.cut(h)) for h in reference]
        r = cn_rouge.get_scores(hypothesis, reference, avg=True)
    elif c=='jp':
        from sumeval.metrics.rouge import RougeCalculator
        rouge = RougeCalculator(stopwords=True, lang="ja")
        print(hypothesis)
        print(reference)
        avg_r = []
        for i in range(len(hypothesis)):
            r = rouge.rouge_l(summary=hypothesis[i],references=reference[i])
            avg_r.append(r)
        r = sum(avg_r)/len(avg_r)

    else:
        r_results = rouge.compute(predictions=hypothesis, references=reference)
        r = r_results['rougeL']
    if c == 'cn':
        b_results = bleu.compute(predictions=hypothesis, references=[[r] for r in reference], max_order=1, smooth=True,tokenizer=tokenizer_zh.TokenizerZh())
    elif c== 'jp':
        b_results = bleu.compute(predictions=hypothesis, references=[[r] for r in reference], max_order=1, smooth=True,tokenizer=tokenizer_ja_mecab.TokenizerJaMecab())
    else:
        b_results = bleu.compute(predictions=hypothesis, references=[[r] for r in reference], max_order=1, smooth=True)
    b = b_results["bleu"]

    print(b)
    print(r)
    print('--------')



