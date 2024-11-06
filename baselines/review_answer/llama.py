import logging
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import json
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict
)
import os
from datasets import load_dataset
import evaluate
rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")
base_path = '/home/dpl944/ProductQA/Bert_classification/train_split'
CUTOFF_LEN = 512
TRAIN_STEPS = 100
BASE_MODEL = "llama-2-7b-hf"
c = 'au'
market_type = 'cross'
model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"

def generate_prompt(data_point):
    return f"""In this task, you will be given a product question, and some reviews. You need to answer the question given the question and reviews.  
### Question:
{data_point["question"]}
### Reviews:
{' '.join(data_point["bm25_top5"])}
### Response:
{data_point["topAnswer"]}"""

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt


train_dataset = load_dataset('json', data_files=os.path.join(base_path,c+'_bm25_review_item_aware_'+market_type+'_market_train.jsonl'), split='train')
val_dataset = load_dataset('json', data_files=os.path.join(base_path,c+'_bm25_review_item_aware_'+market_type+'_market_val.jsonl'), split='train')
for v in train_dataset.features:
    if v not in ['bm25_top5', 'question', 'topAnswer']:
        train_dataset = train_dataset.remove_columns(v)
for v in val_dataset.features:
    if v not in ['bm25_top5', 'question', 'topAnswer']:
        val_dataset = val_dataset.remove_columns(v)
train_data = train_dataset.map(generate_and_tokenize_prompt)
val_data = val_dataset.map(generate_and_tokenize_prompt)


LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
OUTPUT_DIR = "llama_experiments"

model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
resume_from_checkpoint = "" # set this to the adapter_model.bin file you want to resume from

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

if resume_from_checkpoint:
    if os.path.exists(resume_from_checkpoint):
        logging.info(f"Restarting from {resume_from_checkpoint}")
        adapters_weights = torch.load(resume_from_checkpoint)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        logging.info(f"Checkpoint {resume_from_checkpoint} not found")
training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard"
)
data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

model = torch.compile(model)

trainer.train()
model.save_pretrained(OUTPUT_DIR)

# evaluation
logging.info('Start evaluation')
def generate_eval_prompt(data_point):
    eval_prompt = f"""In this task, you will be given a product question, and some reviews. You need to answer the question given the question and reviews.  
    ### Question:
    {data_point["question"]}
    ### Reviews:
    {' '.join(data_point["bm25_top5"])}
    ### Response:
    """
    return eval_prompt

def generate_and_tokenize_eval_prompt(data_point):
    full_prompt = generate_eval_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt, return_tensors="pt")
    return tokenized_full_prompt

model.eval()

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    tokenizer=tokenizer,
    device_map="cuda",
)
test_dataset = [json.loads(i) for i in open(os.path.join(base_path,c+'_bm25_review_item_aware_'+market_type+'_market_test.jsonl')).readlines()]
test_sent = [generate_eval_prompt(i) for i in test_dataset]
actuals = [i['topAnswer'] for i in test_dataset]
from tqdm import tqdm
results = []
with torch.no_grad():
    for t in tqdm(enumerate(test_sent)):
        sequences = pipeline(
            t,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=100,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")
            generated_text = seq['generated_text'][len(t)+1:]
        results.append(generated_text)

r_results = rouge.compute(predictions=results, references=actuals)
b_results = bleu.compute(predictions=results, references=[[r] for r in actuals], max_order=1, smooth=True)
b = b_results["bleu"]
r = r_results['rougeL']
logging.info(f'TEST BLEU score {b}')
logging.info(f'TEST ROUGE score {r}')