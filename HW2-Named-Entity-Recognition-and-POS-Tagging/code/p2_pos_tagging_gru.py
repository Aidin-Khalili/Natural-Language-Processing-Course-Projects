import os
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

os.makedirs('results', exist_ok=True)

# 1. Load the POS Tagging Dataset (just read, no learning)
from datasets import load_dataset
train_data = load_dataset('batterydata/pos_tagging', split='train')
test_data = load_dataset('batterydata/pos_tagging', split='test')

# Get all unique tag names from training set
all_tags = set()
for label_seq in train_data['labels']:
    all_tags.update(label_seq)
label_names = sorted(all_tags)

# 2. Most Frequent POS Tagger (Statistical, no AI)
word_tag_counts = defaultdict(lambda: Counter())
for tokens, tags in zip(train_data['words'], train_data['labels']):
    for w, t in zip(tokens, tags):
        word_tag_counts[w][t] += 1
all_tag_counts = Counter([t for tags in train_data['labels'] for t in tags])
most_common_tag = all_tag_counts.most_common(1)[0][0]

def most_frequent_tagger(tokens):
    tags = []
    for w in tokens:
        if w in word_tag_counts:
            tag = word_tag_counts[w].most_common(1)[0][0]
        else:
            tag = most_common_tag
        tags.append(tag)
    return tags

# Evaluate on test set
true_tags = []
pred_tags = []
for tokens, tags in zip(test_data['words'], test_data['labels']):
    pred = most_frequent_tagger(tokens)
    pred_tags.extend(pred)
    true_tags.extend(tags)
accuracy_heur = accuracy_score(true_tags, pred_tags)
report_heur = classification_report(true_tags, pred_tags, output_dict=True, zero_division=0, labels=label_names)
df_heur = pd.DataFrame(report_heur).transpose()
df_heur.to_csv('results/heuristic_report.csv')
with open('results/heuristic_accuracy.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy_heur:.4f}\n")

# 3. (Optional) Baseline Rule-Based Tagger (Pure heuristics, still NO AI)
import re
def rule_based_tagger(tokens):
    tags = []
    for w in tokens:
        if re.match(r'^[0-9]+$', w):
            tags.append('CD')  # All digits = number
        elif w.lower() in {'the', 'a', 'an'}:
            tags.append('DT')
        elif w.endswith('ly'):
            tags.append('RB')  # Adverb
        elif w.endswith('ing'):
            tags.append('VBG')
        elif w.endswith('ed'):
            tags.append('VBD')
        elif w.istitle():
            tags.append('NNP')  # Proper noun
        elif w in word_tag_counts:
            tag = word_tag_counts[w].most_common(1)[0][0]
            tags.append(tag)
        else:
            tags.append(most_common_tag)
    return tags

# Evaluate rule-based tagger
true_tags_rule = []
pred_tags_rule = []
for tokens, tags in zip(test_data['words'], test_data['labels']):
    pred = rule_based_tagger(tokens)
    pred_tags_rule.extend(pred)
    true_tags_rule.extend(tags)
accuracy_rule = accuracy_score(true_tags_rule, pred_tags_rule)
report_rule = classification_report(true_tags_rule, pred_tags_rule, output_dict=True, zero_division=0, labels=label_names)
df_rule = pd.DataFrame(report_rule).transpose()
df_rule.to_csv('results/rule_based_report.csv')
with open('results/rule_based_accuracy.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy_rule:.4f}\n")

# 4. Plot results
f1_heur = df_heur['f1-score'].loc[label_names]
f1_rule = df_rule['f1-score'].loc[label_names]
plt.figure(figsize=(14,5))
plt.bar(np.arange(len(f1_heur)) - 0.2, f1_heur, width=0.4, label='Most Frequent')
plt.bar(np.arange(len(f1_rule)) + 0.2, f1_rule, width=0.4, label='Rule-based')
plt.xticks(range(len(f1_heur)), label_names, rotation=45)
plt.ylabel('F1 Score')
plt.title('POS Tagging: F1-score Comparison by Tag (No AI)')
plt.legend()
plt.tight_layout()
plt.savefig('results/f1_comparison_noai.png')
plt.close()

print("All steps finished (no AI, only heuristics/rules). Results saved in 'results/' folder.")
