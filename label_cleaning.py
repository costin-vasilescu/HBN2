import re
import pandas as pd

def is_naics3(text):
    return len(text) == 3 and text.isnumeric()

df = pd.read_csv('naics_codes.csv')
df.drop('naics_label', axis=1, inplace=True)
df.dropna(subset=['naics_code'], inplace=True)

df['is_naics3'] = df['naics_code'].apply(is_naics3)
df = df[df['is_naics3'] == True]
df.drop('is_naics3', axis=1, inplace=True)
df.to_csv('cleaned_naics_codes_final.csv', index=False)