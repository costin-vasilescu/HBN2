import pandas as pd
from nltk.stem import PorterStemmer


class PatternMatching:
    def __init__(self, labels):
        names_list = labels['name'].tolist()
        self.transform_lowercase(names_list)
        stemmed_names = self.stem_with_nltk(names_list)
        labels['name'] = stemmed_names
        self.dataframe = labels

    def transform_lowercase(self, sentences_list):
        for i in range(len(sentences_list)):
            sentences_list[i] = sentences_list[i].lower()

    def stem_with_nltk(self, sentences_list):
        stemmer = PorterStemmer()
        stemmed_list = []

        for sentence in sentences_list:
            tokens = sentence.split()
            stemmed_tokens = [stemmer.stem(token) for token in tokens]
            stemmed_sentence = ' '.join(stemmed_tokens)
            stemmed_list.append(stemmed_sentence)

        return stemmed_list

    def predict_naics(self, input):
        # Preprocess input
        input = self.transform_lowercase([input])
        input = self.stem_with_nltk([input])

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