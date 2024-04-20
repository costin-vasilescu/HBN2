import pandas as pd
import nltk
# nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tqdm import trange
import re

def extract_business(dataframe):

    counter = 0

    stemmer = PorterStemmer()

    for i in trange(len(dataframe)):

        company_name = dataframe['commercial_name'][i].lower().split()

        business_name = dataframe['main_business_category'][i].lower().split()
        
        stem_company = [stemmer.stem(word) for word in company_name]

        stem_business = [stemmer.stem(word) for word in business_name]
        
        intersection = list(set(stem_company) & set(stem_business))

        if len(intersection) > 0:
            
            counter += 1
    
    return counter / len(dataframe)


def replace_numbers(dataframe):

    dict_replacements = {
        '2':' to ',
        '3':'e',
        '4':' for ',
        '0': 'o'
    }
    
    for i in trange(len(dataframe)):
        
        company_name = dataframe['commercial_name'][i]
        
        for key in dict_replacements.keys():

            if key in company_name:
                
                try:

                    index = company_name.index(key)

                    if (company_name[index - 1].isalpha() or company_name[index - 1] == " ") and (company_name[index + 1].isalpha() or company_name[index + 1] == " "):
                        
                        modified_name = company_name.replace(key, dict_replacements[key])

                        dataframe.at[i, 'commercial_name'] = modified_name
                
                except IndexError:

                    i += 1
                    
    return dataframe

df = pd.read_csv('tournament_hints_data.csv')

ddf = replace_numbers(df)

print(extract_business(ddf))



