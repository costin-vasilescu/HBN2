from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
from typing import List, Dict
from tqdm import tqdm
import os


def serialize_embeddings(embeddings, filename):
    path = f"embeddings/{filename.split('/')[0]}/"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'embeddings/{filename}.pkl', 'wb') as file:
        pickle.dump(embeddings, file)


def grid_serialize(model_names: List[str], inputs: Dict[str, pd.Series]):
    for model_name in tqdm(model_names, desc='Models'):
        for key, value in tqdm(inputs.items(), desc='Embeddings'):
            model = SentenceTransformer(model_name)
            embeddings = model.encode(value)
            serialize_embeddings(embeddings, f'{model_name}/{key}')


# Load datasets
business_taxonomy = pd.read_csv('Business_category_taxonomy.csv')
naics_taxonomy = pd.read_csv('Naics3labeltaxonomy.csv')
hints = pd.read_csv('tournament_hints_data.csv')

# Prepare data
model_names = ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2',
               'all-distilroberta-v1', 'all-roberta-large-v1',
               'all-mpnet-base-v2']
inputs = {
    'business_label': business_taxonomy['label'],
    'business_description': business_taxonomy['description'],
    'naics_label': naics_taxonomy['naics_label'],
    'naics_description': naics_taxonomy['description'],
    'hints_name': hints['commercial_name'],
    'hints_tags': hints['business_tags'],
    'hints_short_description': hints['short_description'],
    'hints_full_description': hints['description'],
    'hints_category': hints['main_business_category']
}

# Perform grid search
grid_serialize(model_names, inputs)
