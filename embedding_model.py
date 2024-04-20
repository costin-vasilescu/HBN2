from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Embeddings:
    def __init__(self, model_name, targets, files):
        self.model = SentenceTransformer(model_name)
        self.files = files
        self.targets = targets

        # Load embeddings from files
        columns = ['business_label', 'business_description',
                   'naics_label', 'naics_description',
                   'hints_name', 'hints_tags', 'hints_short_description',
                   'hints_full_description', 'hints_category']
        self.embeddings = {}
        for col in columns:
            with open(f'embeddings/{model_name}/{col}.pkl') as file:
                self.embeddings[col] = pickle.load(file)

    def _predict_naics(self, input, target_column):
        input_embedding = self.model.encode(input)
        similarities = cosine_similarity(input_embedding, self.embeddings[target_column])
        closest_idx = np.argmax(similarities)

        if target_column.startswith('naics'):
            pred = self.files['naics']['naics_code'].iloc[closest_idx]
        elif target_column.startswith('hints'):
            pred = self.files['naics']['naics_code'].iloc[closest_idx]

        return pred

    def __call__(self, input, round):
        self._predict_naics(input, target_column=self.targets[round])


targets = {
    1: 'naics_Label',
    2: 'naics_label',
    3: 'naics_description',
    4: 'naics_description',
    5: 'naics_label'
}
business_taxonomy = pd.read_csv('Business_category_taxonomy.csv')
naics_taxonomy = pd.read_csv('Naics3labeltaxonomy.csv')
hints = pd.read_csv('hints.csv')
labels = pd.read_csv('cleaned_naics_codes.csv')
files = {
    'business': business_taxonomy,
    'naics': naics_taxonomy,
    'hints': hints,
    'labels': labels
}








