import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

business_taxonomy = pd.read_csv('Business_category_taxonomy.csv')
naics_taxonomy = pd.read_csv('Naics3labeltaxonomy.csv')
hints = pd.read_csv('tournament_hints_data.csv')
labels = pd.read_csv('cleaned_naics_codes_final.csv')

columns = ['business_label', 'business_description',
           'naics_label', 'naics_description',
           'hints_name', 'hints_tags', 'hints_short_description',
           'hints_full_description', 'hints_category']
model_name = 'all-MiniLM-L6-v2'

columns = ['hints_short_description', 'hints_full_description', 'hints_category']
target_col = ['naics_description', ]
column_dict = {
    'hints_short_description': 'naics_description',
    'hints_full_description': 'naics_description',
    'hints_category': 'naics_label',
    
}
max_hints_idx = len(labels)

embeddings = {}
with open(f'embeddings/{model_name}/{col}.pkl', 'rb') as file:
    embeddings[col] = pickle.load(file)
    if col.startswith('hints'):
        if len(embeddings[col]) > max_hints_idx:
            embeddings[col] = embeddings[col][:max_hints_idx]
with open(f'embeddings/{model_name}/{target_col}.pkl', 'rb') as file:
    embeddings[target_col] = pickle.load(file)
    if target_col.startswith('hints'):
        if len(embeddings[target_col]) > max_hints_idx:
            embeddings[target_col] = embeddings[:max_hints_idx]

similarities = cosine_similarity(embeddings[col], embeddings[target_col])
closest_indices = np.argmax(similarities, axis=1)

result_df = pd.DataFrame(columns=['name', 'naics_code'])
counter = 0
for i, closest_naics_idx in enumerate(closest_indices):
    y_pred = naics_taxonomy['naics_code'].iloc[closest_naics_idx]
    y_true = labels['naics_code'].iloc[i]
    if y_true == y_pred:
        counter += 1
        print(labels['name'].iloc[i], y_true, y_pred)
        result_df = result_df._append({
            'name': labels['name'].iloc[i],
            'naics_code': y_true
        }, ignore_index=True)

print(counter)
result_df.to_csv('correct_predictions.csv', index=False)
