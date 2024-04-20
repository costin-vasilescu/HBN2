import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import pandas as pd


def obtine_business(dataframe):

    counter = 0

    lemmatizer = WordNetLemmatizer()

    for i in range(len(dataframe)):

        flag_synonym = 0

        company_name = dataframe['commercial_name'][i].lower().split()

        business_name = dataframe['main_business_category'][i].lower().split()
        
        lemm_company = [lemmatizer.lemmatize(word) for word in company_name]

        lemm_business = [lemmatizer.lemmatize(word) for word in business_name]
        
        for word in lemm_company:

            word_synset = wn.synsets(word)

            lemmatized_synonyms = [lemma for synset in word_synset for lemma in synset.lemma_names()]

            intersection_synonyms = list(set(lemm_company) & set(lemmatized_synonyms))

            if len(intersection_synonyms) > 0:

                flag_synonym = 1

        intersection_exact_match = list(set(lemm_company) & set(lemm_business))

        if len(intersection_exact_match) > 0 or flag_synonym == 1:
            
            counter += 1
    
    return counter / len(dataframe)

dataframe = pd.read_csv('tournament_hints_data.csv')

print(obtine_business(dataframe))