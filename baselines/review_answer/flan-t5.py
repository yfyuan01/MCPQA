from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import evaluate
import logging
import argparse
from tqdm import tqdm
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")
device = "cuda" if torch.cuda.is_available() else "cpu"
country1 = ['au', 'ca', 'in', 'uk']  # '
country2 = ['br', 'cn', 'fr', 'jp', 'mx']
class AnswergenData(Dataset):
    def __init__(
        self, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = target_text
        self.source_text = source_text

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        # source_text = " ".join(source_text.split())
        # target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }

def train(epoch, tokenizer, model, device, loader, optimizer):

    """
    Function to be called for training with the parameters passed from main function

    """
    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 20 == 0:
            logging.info(f'Training epoch: {epoch}, with loss: {loss}')
            # print(f'Training epoch: {epoch}, with loss: {loss}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(tokenizer, model, device, loader):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask,
              max_length=150,
              num_beams=2,
              repetition_penalty=2.5,
              length_penalty=1.0,
              early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          if _%20==0:
              logging.info(f'Completed {_}')
              # print(f'Completed {_}')
          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals

def T5Trainer(
    text_address, model_params, country, market_type, output_dir
):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging

    logging.info(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    # model = T5ForConditionalGeneration.from_pretrained('t5_outputs_fr/model_files')
    model = model.to(device)
    if args.gpt4:
        train_file = os.path.join(text_address,country + '_gpt4_bm25_review_item_aware_' + market_type + '_market_train.jsonl')
        val_file = os.path.join(text_address, country + '_gpt4_bm25_review_item_aware_' + market_type + '_market_val.jsonl')

    else:
        train_file = os.path.join(text_address,country+'_bm25_review_item_aware_'+market_type+'_market_train.jsonl')
        val_file = os.path.join(text_address, country + '_bm25_review_item_aware_' + market_type + '_market_val.jsonl')
    contents = [json.loads(i) for i in open(train_file).readlines()[:600]] # mark
    if country in country1:
        contents = [i for i in contents if i['topAnswer'] != '']
        source_text_train = [i['question'] + ' '.join(i['bm25_top5']) for i in contents]
        if args.gpt4:
            target_text_train = [i['gpt4_answer'] for i in contents]
        else:
            target_text_train = [i['topAnswer'] for i in contents]
        contents = [json.loads(i) for i in open(val_file).readlines()]
        contents = [i for i in contents if i['topAnswer'] != '']
        source_text_val = [i['question']+' '.join(i['bm25_top5']) for i in contents]
        if args.gpt4:
            target_text_val = [i['gpt4_answer'] for i in contents]
        else:
            target_text_val = [i['topAnswer'] for i in contents]
    elif country in country2:
        if model_params['MODEL'].find('mt5')>0:
            contents = [i for i in contents if i['topAnswer'] != '']
            source_text_train = [i['question'] +' '+ ' '.join(i['bm25_top5']) for i in contents]
            target_text_train = [i['topAnswer'] for i in contents]
            contents = [json.loads(i) for i in open(val_file).readlines()]
            contents = [i for i in contents if i['topAnswer'] != '']
            source_text_val = [i['question'] +' ' + ' '.join(i['bm25_top5']) for i in contents]
            target_text_val = [i['topAnswer'] for i in contents]
        else:
            contents = [i for i in contents if i['topAnswer'] != '']
            if market_type == 'cross' or market_type == 'top3':
                source_text_train = [i['translatedQuestion']+ ' '+' '.join(i['bm25_top5']) for i in contents]
            else:
                source_text_train = [i['translatedQuestion'] + ' ' + ' '.join(i['translated_bm25_top5']) for i in contents]
            if args.gpt4:
                target_text_train = [i['gpt4_answer'] for i in contents]
            else:
                target_text_train = [i['translatedAnswer'] for i in contents]
            contents = [json.loads(i) for i in open(val_file).readlines()]
            contents = [i for i in contents if i['topAnswer'] != '']
            if market_type == 'cross'or market_type == 'top3':
                source_text_val = [i['translatedQuestion'] + ' ' + ' '.join(i['bm25_top5']) for i in contents]
            else:
                source_text_val = [i['translatedQuestion']+ ' ' + ' '.join(i['translated_bm25_top5']) for i in contents]
            if args.gpt4:
                target_text_val = [i['gpt4_answer'] for i in contents]
            else:
                target_text_val = [i['translatedAnswer'] for i in contents]
    print(source_text_train[0])
    print(target_text_train[0])
    logging.info(f'Loaded {len(source_text_train)} training inputs')
    logging.info(f'Loaded {len(source_text_val)} validation inputs')


    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = AnswergenData(
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text_train,
        target_text_train,
    )
    val_set = AnswergenData(
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text_val,
        target_text_val,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    logging.info(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    logging.info(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # evaluating test dataset
    logging.info(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(tokenizer, model, device, val_loader)
        r_results = rouge.compute(predictions=predictions, references=actuals)
        b_results = bleu.compute(predictions=predictions, references=[[r] for r in actuals], max_order=1, smooth=True)
        b = b_results["bleu"]
        r = r_results['rougeL']
        logging.info(f'Validation BLEU score {b}')
        logging.info(f'Validation ROUGE score {r}')


    logging.info(f"[Validation Completed.]\n")
    logging.info(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    logging.info(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")

def T5Tester(text_address, model_params, country, market_type, output_dir="./outputs/"):
    logging.info('Loading model')
    model = T5ForConditionalGeneration.from_pretrained(os.path.join(output_dir,"model_files"))
    model = model.to(device)
    # tokenizer = T5Tokenizer.from_pretrained(os.path.join(output_dir,"model_files"))
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    if args.gpt4:
        test_file = os.path.join(text_address, country + '_gpt4_bm25_review_item_aware_' + market_type + '_market_test.jsonl')
    else:
        test_file = os.path.join(text_address, country + '_bm25_review_item_aware_' + market_type + '_market_test.jsonl')
    contents = [json.loads(i) for i in open(test_file).readlines()]
    contents = [i for i in contents if i['topAnswer']!='']
    if country in country1:
        source_text_test = [i['question']+' '.join(i['bm25_top5']) for i in contents]
        if args.gpt4:
            target_text_test = [i['gpt4_answer'] for i in contents]
        else:
            target_text_test = [i['topAnswer'] for i in contents]
    elif country in country2:
        if model_params['MODEL'].find('mt5') > 0:
            target_text_test = [i['topAnswer'] for i in contents]
            source_text_test = [i['question'] + ' '+ ' '.join(i['bm25_top5']) for i in contents]
        else:
            if args.gpt4:
                target_text_test = [i['gpt4_answer'] for i in contents]
            else:
                target_text_test = [i['translatedAnswer'] for i in contents]
            if market_type == 'single':
                source_text_test = [i['translatedQuestion']+ ' ' +' '.join(i['translated_bm25_top5']) for i in contents]
            else:
                source_text_test = [i['translatedQuestion'] + ' ' + ' '.join(i['bm25_top5']) for i in contents]

    logging.info(f'Loaded {len(source_text_test)} testing inputs')
    testing_set = AnswergenData(
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text_test,
        target_text_test
    )
    test_params = {
        "batch_size": model_params["TEST_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }
    test_loader = DataLoader(testing_set, **test_params)
    logging.info(f"[Start Testing]...\n")
    predictions, actuals = validate(tokenizer, model, device, test_loader)
    r_results = rouge.compute(predictions=predictions, references=actuals)
    b_results = bleu.compute(predictions=predictions, references=[[r] for r in actuals], max_order=1, smooth=True)
    b = b_results["bleu"]
    r = r_results['rougeL']
    logging.info(f'TEST BLEU score {b}')
    logging.info(f'TEST ROUGE score {r}')

model_params = {
    "MODEL": "bigscience/T0_3B",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 2,  # training batch size
    "VALID_BATCH_SIZE": 2,  # validation batch size
    "TEST_BATCH_SIZE": 2,
    "TRAIN_EPOCHS": 10,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 50,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}
parser = argparse.ArgumentParser()
parser.add_argument('--country',type=str)
parser.add_argument('--market',type=str)
parser.add_argument('--gpt4',action='store_true')
args = parser.parse_args()
c = args.country
logging.info(c)
print(c)
market_type = args.market
print(market_type)
logging.info(market_type)
if args.gpt4:
    output_dir = "/projects/nlp_mgr/people/dpl944/gpt4_flant5_outputs_" + c + "_" + market_type
else:
    if model_params['MODEL'].find('mt5')<0:
        output_dir = "/projects/nlp_mgr/people/dpl944/flant5_outputs_"+c+"_"+market_type
    else:
        output_dir = "/projects/nlp_mgr/people/dpl944/mt5_outputs_"+c+"_"+market_type
if args.gpt4:
    text_address = '/home/dpl944/ProductQA/LLMdata/review1000/train_split'
else:
    text_address = "/home/dpl944/ProductQA/Bert_classification/train_split"
T5Trainer(
    text_address=text_address,
    model_params=model_params,
    country=c,
    market_type=market_type,
    output_dir=output_dir
)
T5Tester(
text_address=text_address,
    model_params=model_params,
    country=c,
    market_type=market_type,
    output_dir=output_dir
)