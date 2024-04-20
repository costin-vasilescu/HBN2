import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

with open('embeddings/similarities.pkl', 'rb') as file:
    similarities = pickle.load(file)
with open('embeddings/name_embeddings.pkl', 'rb') as file:
    name_embeddings = pickle.load(file)
with open('embeddings/label_embeddings.pkl', 'rb') as file:
    label_embeddings = pickle.load(file)

naics_taxonomy = pd.read_csv('Naics3labeltaxonomy.csv')
df = pd.read_csv('cleaned_naics_codes.csv')

null_indices = df[df['naics_code'].isnull()].index.tolist()
df.dropna(subset=['naics_code'], inplace=True)
name_embeddings = [item for idx, item in enumerate(name_embeddings) if idx not in null_indices]

# closest_indices = [np.argsort(-row)[0] for row in similarities]
similarity_matrix = cosine_similarity(name_embeddings, label_embeddings)
closest_indices = np.argmax(similarity_matrix, axis=1)

y_pred = []
y_true = []
for i, max_idx in tqdm(enumerate(closest_indices)):
    y_pred.append(naics_taxonomy['naics_code'].iloc[max_idx])
    y_true.append(df['naics_code'].iloc[i])

print(accuracy_score(y_true, y_pred))



