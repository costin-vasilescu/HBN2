import re
import pandas as pd

def find_consecutive_digits(text):
    # Search for three consecutive digits in the text
    match = re.search(r'\d{3}', str(text))
    # Return the matched group if found, otherwise return None
    return match.group() if match else None

df = pd.read_csv('naics_codes_stable.csv')
df['naics_code'] = df['naics_code'].apply(find_consecutive_digits)
df.to_csv('cleaned_naics_codes.csv', index=False)