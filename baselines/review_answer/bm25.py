# This is the generation result for the bm25 generated top-1 review
# from rouge_score import rouge_scorer
import nltk
import json
import evaluate
# from nltk.tokenize import word_tokenize
country1 = ['au','ca','in','uk'] #'
# country2 = ['br','fr','mx']
# country3 = ['cn','jp']
country2 = ['br','cn','fr','jp','mx']
rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
hypothesis = []
reference = []
for c in country1:
    print(c)
    filename = '/home/dpl944/ProductQA/Bert_classification/train_split/'+c+'_bm25_review_item_aware_cross_market_test.jsonl'
    # filename = '/home/dpl944/ProductQA/LLMdata/review1000/train_split/' + c + '_bm25_review_item_aware_single_market_test.jsonl'
    contents = [json.loads(i) for i in open(filename)]
    contents = [i for i in contents if i['topAnswer']!='']
    hypothesis = [i['bm25_top5'][0] for i in contents]
    reference = [i['topAnswer'] for i in contents] #mark
    r_results = rouge.compute(predictions=hypothesis, references=reference)
    b_results = bleu.compute(predictions=hypothesis, references=[[r] for r in reference], max_order=1, smooth=True)
    bert_results = bertscore.compute(predictions=hypothesis, references=reference, model_type="distilbert-base-uncased")
    b = b_results["bleu"]
    r = r_results['rougeL']
    bscore = sum(bert_results['f1']) / len(bert_results['f1'])
    print(b)
    print(r)
    print(bscore)
    print('--------')
#     # break


for c in country2:
    print(c)
    filename = '/home/dpl944/ProductQA/Bert_classification/train_split/'+c+'_bm25_review_item_aware_cross_market_test.jsonl'
#     # filename = '/home/dpl944/ProductQA/LLMdata/review1000/results_'+c+'.jsonl'
#     filename = '/home/dpl944/ProductQA/LLMdata/review1000/train_split/' + c + '_bm25_review_item_aware_single_market_test.jsonl'
    contents = [json.loads(i) for i in open(filename)]
    contents = [i for i in contents if i['translatedAnswer']!='']
    hypothesis = [i['bm25_top5'][0] for i in contents] #translated_
    reference = [i['topAnswer'] for i in contents]
    r_results = rouge.compute(predictions=hypothesis, references=reference)
    b_results = bleu.compute(predictions=hypothesis, references=[[r] for r in reference], max_order=1, smooth=True)
    bert_results = bertscore.compute(predictions=hypothesis, references=reference, model_type="distilbert-base-uncased")
    b = b_results["bleu"]
    r = r_results['rougeL']
    print(b)
    print(r)
    print(bscore)
    print('--------')

# for c in country2:
#     print(c)
#     filename = '/home/dpl944/ProductQA/Bert_classification/train_split/' + c + '_bm25_review_item_aware_cross_market_test.jsonl'
#     contents = [json.loads(i) for i in open(filename)]
#     contents = [i for i in contents if i['translatedAnswer']!='']
#     hypothesis += [i['bm25_top5'][0] for i in contents]
#     reference += [i['translatedAnswer'] for i in contents]
# r_results = rouge.compute(predictions=hypothesis, references=reference)
# b_results = bleu.compute(predictions=hypothesis, references=[[r] for r in reference], max_order=1, smooth=True)
# b = b_results["bleu"]
# r = r_results['rougeL']
# print(b)
# print(r)