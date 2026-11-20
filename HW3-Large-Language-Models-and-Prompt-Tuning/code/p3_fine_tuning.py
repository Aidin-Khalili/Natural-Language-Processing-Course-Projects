import os
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set up paths
DRIVE_FOLDER_PATH = "/home/Aidin/"
dataset_zip_path = os.path.join(DRIVE_FOLDER_PATH, "dataset.zip")
dataset_extract_path = os.path.join(DRIVE_FOLDER_PATH, "dataset")

# 1. Unzip if needed
if not os.path.exists(dataset_extract_path):
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_extract_path)

def load_and_normalize(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    return df

train_path = os.path.join(dataset_extract_path, "train.csv")
val_path = os.path.join(dataset_extract_path, "val.csv")
test_path = os.path.join(dataset_extract_path, "test.csv")

train_df = load_and_normalize(train_path)
val_df = load_and_normalize(val_path)
test_df = load_and_normalize(test_path)

def map_label(rating):
    if rating in [4, 5]:
        return 2  # positive
    elif rating == 3:
        return 1  # neutral
    else:
        return 0  # negative

for df in [train_df, val_df, test_df]:
    df['label'] = df['label'].apply(map_label)

# Use distilgpt2 for better speed and support
MODEL_NAME = 'distilgpt2'
MAX_LEN = 64
BATCH_SIZE = 2
EPOCHS = 2
LR = 3e-5
LABEL2WORD = {0: "negative", 1: "neutral", 2: "positive"}
WORD2LABEL = {v: k for k, v in LABEL2WORD.items()}
INSTRUCTION = "Review: {}\nIs this review positive, neutral, or negative? Sentiment:"

class GPT2SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, prompt=INSTRUCTION):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prompt = prompt
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.prompt.format(str(self.texts[idx]))
        label_word = LABEL2WORD[self.labels[idx]]
        target = " " + label_word
        max_input = self.max_len - len(self.tokenizer.encode(target))
        input_ids = self.tokenizer.encode(text, truncation=True, max_length=max_input, return_tensors="pt").squeeze(0)
        target_ids = self.tokenizer.encode(target, return_tensors="pt").squeeze(0)
        labels = torch.ones(input_ids.shape[0] + target_ids.shape[0], dtype=torch.long) * -100
        labels[-target_ids.shape[0]:] = target_ids
        input_ids = torch.cat([input_ids, target_ids])
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

train_ds = GPT2SentimentDataset(train_df, tokenizer, MAX_LEN)
val_ds = GPT2SentimentDataset(val_df, tokenizer, MAX_LEN)
test_ds = GPT2SentimentDataset(test_df, tokenizer, MAX_LEN)

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))
model = model.to('cpu')
optimizer = AdamW(model.parameters(), lr=LR)

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Training", leave=False)
    for batch in loop:
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    loop = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in loop:
            input_ids = batch['input_ids']
            labels = batch['labels']
            attention_mask = batch['attention_mask']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def generate_sentiment(model, text, prompt=INSTRUCTION, max_len=MAX_LEN):
    input_text = prompt.format(text)
    max_input_len = 61
    input_ids = tokenizer.encode(input_text, truncation=True, max_length=max_input_len, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 3,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=torch.ones_like(input_ids),
        do_sample=False
    )
    gen_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:]).strip().split(" ")[0].lower()
    if "pos" in gen_text:
        return 2
    elif "neu" in gen_text:
        return 1
    elif "neg" in gen_text:
        return 0
    else:
        return -1

print("Starting training...")
best_val_loss = float('inf')
best_model_state = None
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        torch.save(best_model_state, dataset_extract_path + "best_gpt2_sentiment.pt")
    if epoch > 1 and val_loss > best_val_loss:
        print("Early stopping.")
        break

model.load_state_dict(torch.load(dataset_extract_path + "best_gpt2_sentiment.pt"))

test_preds = []
test_labels = []
for i in tqdm(range(len(test_df)), desc="Test inference"):
    text = test_df.iloc[i]['text']
    label = test_df.iloc[i]['label']
    pred = generate_sentiment(model, text)
    test_preds.append(pred)
    test_labels.append(label)

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)

def accuracy(preds, labels):
    valid = preds != -1
    return (preds[valid] == labels[valid]).mean() if np.any(valid) else 0.0

def precision_recall_f1(preds, labels, num_classes=3):
    prec, rec, f1 = [], [], []
    for cls in range(num_classes):
        tp = np.sum((preds == cls) & (labels == cls))
        fp = np.sum((preds == cls) & (labels != cls))
        fn = np.sum((preds != cls) & (labels == cls))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        prec.append(p)
        rec.append(r)
        f1.append(f)
    return prec, rec, f1

test_acc = accuracy(test_preds, test_labels)
test_prec, test_rec, test_f1 = precision_recall_f1(test_preds, test_labels)

print(f"Test Accuracy: {test_acc:.4f}")
for i, label in enumerate(["Negative", "Neutral", "Positive"]):
    print(f"{label}: Precision={test_prec[i]:.4f} Recall={test_rec[i]:.4f} F1={test_f1[i]:.4f}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, test_preds, labels=[0,1,2])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Neu", "Pos"], yticklabels=["Neg", "Neu", "Pos"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Decoder Only Model (Prompt Improved)")
plt.show()
