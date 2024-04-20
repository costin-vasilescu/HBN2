import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load embeddings
with open('embeddings/name_embeddings.pkl', 'rb') as file:
    name_embeddings = pickle.load(file)

# Load data
naics_taxonomy = pd.read_csv('Naics3labeltaxonomy.csv')
df = pd.read_csv('cleaned_naics_codes.csv')

# Preprocess to remove entries without NAICS codes
null_indices = df[df['naics_code'].isnull()].index.tolist()
df.dropna(subset=['naics_code'], inplace=True)
name_embeddings = np.array([item for idx, item in enumerate(name_embeddings) if idx not in null_indices])

# Apply indexing using numpy arrays
indices = list(range(len(df)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_embeddings = name_embeddings[train_idx]
test_embeddings = name_embeddings[test_idx]

# Compute cosine similarity in batch
similarity_matrix = cosine_similarity(test_embeddings, train_embeddings)
closest_indices = np.argmax(similarity_matrix, axis=1)

# Map indices to actual predictions
y_pred = []
y_true = []
for i, pred in tqdm(enumerate(closest_indices)):
    y_pred.append(df['naics_code'].iloc[train_idx[pred]])
    y_true.append(df['naics_code'].iloc[test_idx[i]])

# Compute accuracy
print(accuracy_score(y_true, y_pred))