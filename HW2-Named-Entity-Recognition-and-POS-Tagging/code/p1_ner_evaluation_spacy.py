import os
import json
import zipfile
import random
import string
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy.util import minibatch
from spacy.training import Example
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore", category=UserWarning)

# Define constants
DATA_ZIP_PATH = "/home/Aidin/resume_ner.zip"
EXTRACT_DIR = "resume_data"
MODEL_OUTPUT_DIR = "output/ner_resume_model"
EVAL_CSV_PATH = "output/ner_evaluation_metrics.csv"
HEATMAP_PATH = "output/ner_evaluation_heatmap.png"
TRAIN_ITERATIONS = 60
BATCH_SIZE = 4
DROPOUT = 0.35
LABELS = ["College Name", "Location", "Designation", "Email Address", "Name", "Skills"]

# Ensure necessary directories exist
os.makedirs("output", exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

# Extract the dataset
with zipfile.ZipFile(DATA_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

# Utility functions for entity cleaning

def clean_span(text, start, end):
    while start < end and text[start] in string.whitespace + string.punctuation:
        start += 1
    while end > start and text[end - 1] in string.whitespace + string.punctuation:
        end -= 1
    return start, end

def remove_overlaps(entities):
    entities = sorted(entities, key=lambda x: (x[0], -x[1]))
    result = []
    last_end = -1
    for s, e, label in entities:
        if s >= last_end:
            result.append((s, e, label))
            last_end = e
    return result

# Load and preprocess data
json_path = os.path.join(EXTRACT_DIR, "Entity Recognition in Resumes.json")
with open(json_path, 'r', encoding='utf-8') as file:
    raw_data = [json.loads(line) for line in file]

data_for_spacy = []
for entry in raw_data:
    if not entry.get("annotation"):
        continue
    text = entry["content"]
    ents = []
    for ann in entry["annotation"]:
        if not ann.get("label") or not ann.get("points"):
            continue
        label = ann["label"][0]
        for point in ann["points"]:
            start, end = clean_span(text, point["start"], point["end"] + 1)
            if start < end:
                ents.append((start, end, label))
    ents = remove_overlaps(ents)
    if ents:
        data_for_spacy.append((text, {"entities": ents}))

# Shuffle and split data
random.shuffle(data_for_spacy)
train_size = int(0.8 * len(data_for_spacy))
train_data = data_for_spacy[:train_size]
test_data = data_for_spacy[train_size:]

# Create and train the model
nlp = spacy.blank("en")
ner_pipe = nlp.add_pipe("ner")

for _, ann in train_data:
    for start, end, label in ann["entities"]:
        ner_pipe.add_label(label)

optimizer = nlp.begin_training()
for i in range(TRAIN_ITERATIONS):
    random.shuffle(train_data)
    losses = {}
    batches = minibatch(train_data, size=BATCH_SIZE)
    for batch in batches:
        examples = [Example.from_dict(nlp.make_doc(text), annot) for text, annot in batch]
        nlp.update(examples, drop=DROPOUT, losses=losses)
    print(f"Iteration {i+1}: Loss = {losses['ner']:.4f}")

# Save the trained model
nlp.to_disk(MODEL_OUTPUT_DIR)

# Evaluate the model
def compute_metrics(true, pred, labels):
    results = []
    for label in labels:
        yt = [l == label for l in true]
        yp = [l == label for l in pred]
        p, r, f1, _ = precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)
        results.append({"Recognized Entity": label, "Precision": p, "Recall": r, "F-Score": f1})
    return pd.DataFrame(results)

y_true, y_pred = [], []
for text, ann in test_data:
    doc = nlp(text)
    pred_labels = {ent.label_ for ent in doc.ents}
    true_labels = {label for _, _, label in ann["entities"]}
    for label in LABELS:
        y_true.append(label if label in true_labels else None)
        y_pred.append(label if label in pred_labels else None)

df_metrics = compute_metrics(y_true, y_pred, LABELS)
df_metrics.to_csv(EVAL_CSV_PATH, index=False)

plt.figure(figsize=(10, 6))
sns.heatmap(df_metrics.set_index("Recognized Entity")[["F-Score", "Recall", "Precision"]], annot=True, fmt=".2f")
plt.title("NER Evaluation Metrics")
plt.tight_layout()
plt.savefig(HEATMAP_PATH)