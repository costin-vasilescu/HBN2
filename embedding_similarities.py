from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')
naics_taxonomy = pd.read_csv('Naics3labeltaxonomy.csv')
df = pd.read_csv('cleaned_naics_codes.csv')
label_mapping = dict(zip(naics_taxonomy['naics_label'], naics_taxonomy['naics_code']))

# Test pickling embeddings
# test = model.encode(df['name'].iloc[:2])
# with open('embeddings/test_embeddings.pkl', 'wb') as file:
#     pickle.dump(test, file)
# with open('embeddings/test_embeddings.pkl', 'rb') as file:
#     test_embeddings = pickle.load(file)
# print(test_embeddings)

# Generate embeddings
label_embeddings = model.encode(naics_taxonomy['naics_label'])
with open('embeddings/label_embeddings.pkl', 'wb') as file:
    pickle.dump(label_embeddings, file)
name_embeddings = model.encode(df['name'])
with open('embeddings/name_embeddings.pkl', 'wb') as file:
    pickle.dump(name_embeddings, file)
similarities = [cosine_similarity([e1], [e2])[0][0] for e1, e2 in zip(name_embeddings, label_embeddings)]
with open('embeddings/similarities.pkl', 'wb') as file:
    pickle.dump(similarities, file)