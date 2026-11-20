import os
import zipfile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns

DRIVE_FOLDER_PATH = "/home/Aidin/"
dataset_zip_path = os.path.join(DRIVE_FOLDER_PATH, "dataset.zip")

def load_and_normalize_from_zip(zip_path):
    # Open the zip file and read CSVs directly into memory
    data = {}
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for split in ['train.csv', 'val.csv', 'test.csv']:
            with zip_ref.open(split) as f:
                df = pd.read_csv(f)
                df.columns = df.columns.str.lower()
                data[split.replace('.csv', '')] = df
    return data['train'], data['val'], data['test']

train_df, val_df, test_df = load_and_normalize_from_zip(dataset_zip_path)

print(f"Train shape: {train_df.shape}")
print(f"Val shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")

# Preprocess labels
def map_label(rating):
    if rating in [4, 5]:
        return 2  # positive
    elif rating == 3:
        return 1  # neutral
    else:
        return 0  # negative

for df in [train_df, val_df, test_df]:
    df['label'] = df['label'].apply(map_label)

print("Train label distribution:\n", train_df['label'].value_counts())
print("Val label distribution:\n", val_df['label'].value_counts())
print("Test label distribution:\n", test_df['label'].value_counts())

# ---- Soft Prompt Tuning (in-memory) ----

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PRETRAINED_MODEL = 'bert-base-uncased'
BATCH_SIZE = 32
MAX_LEN = 128
NUM_CLASSES = 3
SOFT_PROMPT_TOKENS = 10
EPOCHS = 10

# Dataset Class
class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
            add_special_tokens=True
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
train_ds = ReviewDataset(train_df, tokenizer, MAX_LEN)
val_ds = ReviewDataset(val_df, tokenizer, MAX_LEN)
test_ds = ReviewDataset(test_df, tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Model
class SoftPromptBERT(nn.Module):
    def __init__(self, bert_model_name, num_classes, soft_prompt_tokens=10):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.hidden_size = self.bert.config.hidden_size
        self.soft_prompts = nn.Parameter(torch.randn(soft_prompt_tokens, self.hidden_size))
        self.classifier = nn.Linear(self.hidden_size, num_classes)
    def forward(self, input_ids, attention_mask, token_type_ids):
        batch_size = input_ids.size(0)
        device = input_ids.device
        inputs_embeds = self.bert.embeddings(input_ids=input_ids)
        soft_prompt = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        concat_embeds = torch.cat([soft_prompt, inputs_embeds], dim=1)
        soft_attention = torch.ones((batch_size, self.soft_prompts.shape[0]), dtype=attention_mask.dtype, device=device)
        concat_attention_mask = torch.cat([soft_attention, attention_mask], dim=1)
        outputs = self.bert(
            inputs_embeds=concat_embeds,
            attention_mask=concat_attention_mask,
            token_type_ids=None
        )
        cls_output = outputs.last_hidden_state[:, self.soft_prompts.shape[0], :]
        logits = self.classifier(cls_output)
        return logits

model = SoftPromptBERT(PRETRAINED_MODEL, NUM_CLASSES, SOFT_PROMPT_TOKENS).to(DEVICE)
optimizer = torch.optim.AdamW([
    {'params': model.soft_prompts},
    {'params': model.classifier.parameters()}
], lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)

# Training Loop
best_val_loss = float('inf')
best_model_state = None
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        torch.save(best_model_state, os.path.join(DRIVE_FOLDER_PATH, 'best_soft_prompt_model.pt'))
    if epoch > 1 and val_loss > best_val_loss:
        print("Early stopping triggered.")
        break

# Test Evaluation
model.load_state_dict(torch.load(os.path.join(DRIVE_FOLDER_PATH, 'best_soft_prompt_model.pt')))
_, test_preds, test_labels = evaluate(model, test_loader, criterion)

def accuracy(preds, labels):
    return (preds == labels).mean()

def precision_recall_f1(preds, labels, num_classes=3):
    prec = []
    rec = []
    f1 = []
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
test_prec, test_rec, test_f1 = precision_recall_f1(test_preds, test_labels, num_classes=NUM_CLASSES)

print(f"Test Accuracy: {test_acc:.4f}")
for i, label in enumerate(["Negative", "Neutral", "Positive"]):
    print(f"{label}: Precision={test_prec[i]:.4f} Recall={test_rec[i]:.4f} F1={test_f1[i]:.4f}")

# ---- Confusion Matrix (from scratch) and Heatmap ----
def confusion_matrix(y_true, y_pred, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

cm = confusion_matrix(test_labels, test_preds, num_classes=NUM_CLASSES)
print("Confusion Matrix:\n", cm)

# Plot Confusion Matrix
labels = ["Negative", "Neutral", "Positive"]
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Soft Prompt BERT')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
cm_path = os.path.join(DRIVE_FOLDER_PATH, 'soft_prompt_confusion_matrix.png')
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix heatmap saved at {cm_path}")
