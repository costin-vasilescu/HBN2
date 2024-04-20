import pandas as pd
from nltk.stem import PorterStemmer

# Define the file path
file_path = 'cleaned_naics_codes_V2.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Take user input

def transform_lowercase(sentences_list):
    for i in range(len(sentences_list)):
        sentences_list[i] = sentences_list[i].lower()

def stem_with_nltk(sentences_list):
    stemmer = PorterStemmer()
    stemmed_list = []

    for sentence in sentences_list:
        tokens = sentence.split()
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        stemmed_sentence = ' '.join(stemmed_tokens)
        stemmed_list.append(stemmed_sentence)

    return stemmed_list


names_list = df['name'].tolist()


transform_lowercase(names_list)

stemmed_names = stem_with_nltk(names_list)

# Update the 'name' column in the DataFrame with the stemmed names
df['name'] = stemmed_names

class Game:

    def __init__(self, dataframe):

        self.dataframe = dataframe
    
    def predict_naics(self, input):
        
        
        substrings = set()
        for i in range(len(input) - 2):
            substring = input[i:i + 3]
            substrings.add(substring)
        max=0
        imax = -1
        for word in self.dataframe['name']:
            count=0
            for subs in substrings:
                count = count+ word.count(subs)
            if count>max:
                max=count
                imax=word
        
        if imax == -1:
            return None
        
        filtered_df = self.dataframe[self.dataframe['name'] == imax]
        value_column2 = filtered_df.iloc[0]['naics_code']

        return value_column2

patt_match = Game(df)
comp = 'HQ Machine Tech'
transform_lowercase([comp])

stemmed_string = stem_with_nltk([comp])

print(patt_match.predict_naics(stemmed_string))
