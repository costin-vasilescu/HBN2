import pandas as pd
from nltk.stem import SnowballStemmer
import nltk


class PatternMatching:
    def __init__(self, labels):
        self.stemmer = SnowballStemmer('english')
        labels['name'] = labels['name'].apply(self.preprocess_text)
        self.dataframe = labels

    def preprocess_text(self, input_text):
        # Example preprocessing function
        tokens = nltk.word_tokenize(input_text.lower())
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)

    def predict_naics(self, input):
        # Preprocess input
        input = self.preprocess_text(input)

        substrings = set()
        for i in range(len(input) - 2):
            substring = input[i:i + 3]
            substrings.add(substring)
        max = 0
        imax = -1
        for word in self.dataframe['name']:
            count = 0
            for subs in substrings:
                count = count + word.count(subs)
            if count > max:
                max = count
                imax = word

        if imax == -1:
            return None

        filtered_df = self.dataframe[self.dataframe['name'] == imax]
        value_column2 = filtered_df.iloc[0]['naics_code']

        return value_column2