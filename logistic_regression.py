from joblib import load
import re
import nltk
from nltk.stem import SnowballStemmer
from joblib import load

nltk.download('punkt')
nltk.download('wordnet')


class LR:
    def __init__(self, model_path):
        # Load the trained pipeline
        self.pipeline = load(model_path)
        self.stemmer = SnowballStemmer('english')

    def preprocess_text(self, input_text):
        # Example preprocessing function
        tokens = nltk.word_tokenize(input_text.lower())
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)

    def predict(self, input_text):
        # Predict using the loaded pipeline, which handles vectorization automatically
        prediction = self.pipeline.predict([self.preprocess_text(input_text)])
        return prediction

    def predict_proba(self, input_text):
        # Predict probabilities using the loaded pipeline
        probabilities = self.pipeline.predict_proba([self.preprocess_text(input_text)])
        return probabilities