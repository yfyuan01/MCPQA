import json
import os
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from torch.utils.data import DataLoader
import random
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
train_batch_size=16
country = ['cn','jp','br','mx','fr']
country1 = ['uk','au','in','ca']
market = 'cross'
base_address = '/home/dpl944/ProductQA/LLMdata/ques200'
for c in country:
    print(c)
    main_data_file = os.path.join(base_address,'results_'+c+'.jsonl')
    main_data = [json.loads(i) for i in open(main_data_file).readlines()]
    train_examples = []
    for m in main_data[:700]:
        p = [m['translatedQuestion']+' '+m['translatedAnswer'],m['relatedQuestion']+' '+m['relatedAnswer']]
        train_examples.append(InputExample(texts=p,label=int(m['gpt4_score'])))
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=3)
    epochs = 10
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        output_path='trained_models/bert_test_'+market+'_'+c+'_multilingual',
        show_progress_bar=True,
    )

for c in country1:
    print(c)
    main_data_file = os.path.join(base_address,'results_'+c+'.jsonl')
    main_data = [json.loads(i) for i in open(main_data_file).readlines()]
    train_examples = []
    for m in main_data[:700]:
        p = [m['question']+' '+m['topAnswer'],m['relatedQuestion']+' '+m['relatedAnswer']]
        train_examples.append(InputExample(texts=p,label=int(m['gpt4_score'])))
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=3)
    epochs = 10
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        output_path='trained_models/bert_test_'+market+'_'+c+'_multilingual',
        show_progress_bar=True,
    )
