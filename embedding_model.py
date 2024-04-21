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
        max_hints_idx = len(files['labels'])
        self.embeddings = {}
        for col in columns:
            with open(f'embeddings/{model_name}/{col}.pkl', 'rb') as file:
                self.embeddings[col] = pickle.load(file)
            if col.startswith('hints'):
                if len(self.embeddings[col]) > max_hints_idx:
                    self.embeddings[col] = self.embeddings[col][:max_hints_idx]

    def _predict_naics(self, input, target_column):
        input_embedding = self.model.encode(input)
        similarities = cosine_similarity(input_embedding.reshape(1,-1), self.embeddings[target_column])
        closest_idx = np.argmax(similarities, axis=1)

        if target_column.startswith('naics'):
            file = self.files['naics']
        elif target_column.startswith('hints'):
            file = self.files['labels']
        pred = file['naics_code'].iloc[closest_idx]

        return pred.values[0]

    def __call__(self, input, round):
        return self._predict_naics(input, target_column=self.targets[round])


# targets = {
#     1: 'hints_category',
#     2: 'hints_category',
#     3: 'naics_description',
#     4: 'naics_description',
#     5: 'hints_tags'
# }
# business_taxonomy = pd.read_csv('Business_category_taxonomy.csv')
# naics_taxonomy = pd.read_csv('Naics3labeltaxonomy.csv')
# hints = pd.read_csv('tournament_hints_data.csv')
# labels = pd.read_csv('cleaned_naics_codes.csv')
# files = {
#     'business': business_taxonomy,
#     'naics': naics_taxonomy,
#     'hints': hints,
#     'labels': labels
# }
#
# model_name = 'all-mpnet-base-v2'
# embeddings = Embeddings(model_name, targets, files)
# print(embeddings('Designing Faces', 1))







