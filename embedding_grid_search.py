import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm, trange
import pickle
import numpy as np
import json

# Load files
# business_taxonomy = pd.read_csv('Business_category_taxonomy.csv')
# naics_taxonomy = pd.read_csv('Naics3labeltaxonomy.csv')
# hints = pd.read_csv('tournament_hints_data.csv')
labels = pd.read_csv('cleaned_naics_codes.csv')['naics_code']
max_hints_idx = len(labels)

# Split data
columns = ['business_label', 'business_description',
           'naics_label', 'naics_description',
           'hints_name', 'hints_tags', 'hints_short_description',
           'hints_full_description', 'hints_category']

model_name = 'all-MiniLM-L6-v2'
embeddings = {}
indices = {}
for col in columns:
    with open(f'embeddings/{model_name}/{col}.pkl', 'rb') as file:
        embeddings[col] = pickle.load(file)
    if col.startswith('hints'):
        if len(embeddings[col]) > max_hints_idx:
            embeddings[col] = embeddings[col][:max_hints_idx]

    idx_list = list(range(len(embeddings[col])))
    train_idx, test_idx = train_test_split(idx_list, test_size=0.2, random_state=42)
    indices[col] = {'train': train_idx, 'test': test_idx}

# Testing
scores = {}
round_targets = {
    1: ['naics_label', 'naics_description', 'hints_tags', 'hints_short_description', 'hints_full_description', 'hints_category'],
    2: ['naics_label', 'naics_description', 'hints_short_description', 'hints_full_description', 'hints_category'],
    3: ['naics_label', 'naics_description', 'hints_tags', 'hints_full_description', 'hints_category'],
    4: ['naics_label', 'naics_description', 'hints_tags', 'hints_short_description', 'hints_category'],
    5: ['naics_label', 'naics_description', 'hints_tags', 'hints_short_description', 'hints_full_description']
}

for r in trange(5, desc='Rounds', leave=True):
    target_scores = {}
    for target in tqdm(round_targets[r+1], desc='Columns', leave=False):
        train_idx = indices[target]['train']
        test_idx = indices[target]['test']
        train_embeddings = embeddings[target][train_idx]
        test_embeddings = embeddings[target][test_idx]

        similarities = cosine_similarity(test_embeddings, train_embeddings)
        closest_indices = np.argmax(similarities, axis=1)

        y_pred = []
        y_true = []
        for i, closest_train_idx in enumerate(closest_indices):
            y_pred.append(labels[closest_train_idx])
            y_true.append(labels[test_idx[i]])

        target_scores[target] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1': f1_score(y_true, y_pred, average='macro'),
            'report': classification_report(y_true, y_pred)
        }
        print(f'---Round: {r+1}, Target: {target}---')
        print(f"accuracy: {target_scores[target]['accuracy']}")
        print(f"precision: {target_scores[target]['precision']}")
        print(f"recall: {target_scores[target]['recall']}")
        print(f"f1: {target_scores[target]['f1']}")
    scores[r] = target_scores

with open(f'{model_name}_scores.json', 'w') as file:
    json.dump(file)
























