import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from tqdm.auto import tqdm
import numpy as np
import random
import os

CFG = {
    'MODEL_NAME': "beomi/OPEN-SOLAR-KO-10.7B",
    'LEARNING_RATE': 1e-4,
    'STEPS': 2000,
    'SEED': 42,
    'exp_name': 'exp36'
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(CFG['SEED'])


train_df = pd.read_csv("../data/train.csv")
rearranged_train_df = pd.DataFrame()
for i in tqdm(range(len(train_df))):
    for j in range(4):
        idx = train_df.loc[i, f"answer_{j}"]
        rearranged_train_df.loc[i, f"sentence_{j}"] = train_df.iloc[i, idx+1]
for i in range(4):
    rearranged_train_df[f'answer_{i}'] = i

def augment_data_fixed(df):
    random.seed(42)
    augmented_data = []
    for i in range(len(df)):
        original_row = df.iloc[i].to_dict()
        augmented_data.append(original_row)
        sentence_list = []
        for j in range(4):
            sentence_list.append(df.iloc[i][f'sentence_{j}'])
        for k in range(5):
            used_indices = random.sample(range(4), 4)
            new_row = {}
            for j in range(4):
                new_row[f'sentence_{j}'] = sentence_list[used_indices[j]]
            for j in range(4):
                new_row[f'answer_{j}'] = used_indices.index(j)
            augmented_data.append(new_row)
    return pd.DataFrame(augmented_data)

augmented_df = augment_data_fixed(rearranged_train_df)
augmented_df = augmented_df.drop(augmented_df.index[::4]).reset_index(drop=True)
train = augmented_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
def make_input(row):
    sentences = [row[f"sentence_{i}"] for i in range(4)]
    input_text = "문장을 순서대로 정렬하세요: " + " </s> ".join(sentences)
    answer = [row[f"answer_{i}"] for i in range(4)]
    target_text = " ".join(map(str, answer))
    return {"input": input_text, "target": target_text}

inputs = train.apply(make_input, axis=1).tolist()

def compute_accuracy(preds, labels):
    correct = 0
    total = 0
    for p, l in zip(preds, labels):
        if p == l:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

def predict_order(sent_list):
    prompt = sent_list + "\n정답:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=512
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("정답:")[-1].strip().replace('\n', ' ')
    try:
        order = list(map(int, answer.strip().split()))
        return order
    except ValueError as e:
        return [0, 1, 2, 3]

train_data = inputs
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name= CFG['MODEL_NAME'],
    max_seq_length=4096,
    dtype=None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

def tokenize(batch):
    prompts = [inp.strip() + "\n정답:" for inp in batch["input"]]
    full_texts = [p + " " + tgt + tokenizer.eos_token
                for p, tgt in zip(prompts, batch["target"])]
    tok = tokenizer(full_texts,
                    max_length=512,
                    truncation=True,
                    padding="max_length")
    labels = []
    for prompt, ids in zip(prompts, tok["input_ids"]):
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        lbl = ids.copy()
        lbl[:prompt_len] = [-100] * prompt_len
        labels.append(lbl)
    tok["labels"] = labels
    return tok

tokenized_train = train_dataset.map(tokenize, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_train,
    dataset_num_proc=8,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        logging_steps=1,
        learning_rate=CFG['LEARNING_RATE'],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=CFG['SEED'],
        output_dir=f"../model/{CFG['exp_name']}",
    ),
)
trainer.train()

model.eval()

test = pd.read_csv("../data/test.csv")
def make_input(row):
    sentences = [row[f"sentence_{i}"] for i in range(4)]
    input_text = "문장을 순서대로 정렬하세요: " + " </s> ".join(sentences)
    return input_text

test_inputs = test.apply(make_input, axis=1).tolist()

def predict_order(sent_list):
    prompt = sent_list + "\n정답:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=512
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("정답:")[-1].strip().replace('\n', ' ')
    try:
        order = list(map(int, answer.strip().split()))
        return order
    except ValueError as e:
        return [0, 1, 2, 3]

predictions = []
for sent_group in tqdm(test_inputs, desc="Predicting"):
    pred = predict_order(sent_group)
    predictions.append(pred)

sample_submission = pd.read_csv("../data/sample_submission.csv")
for i in range(4):
    sample_submission[f"answer_{i}"] = [
        pred[i] if len(pred) == 4 else i for pred in predictions
    ]
sample_submission.to_csv(f"../submission/{CFG['exp_name']}.csv", index=False)