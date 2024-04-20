from openai import OpenAI
import os
import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

prompt_template = """Consider the following data about a company:
name: {name}
business_tags: {tags}
description: {description}
category: {category}

Generate the NAICS3 code for this company. Don't write anything else in your answer.
NAICS3 code:"""

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Load the dataset
df = pd.read_csv('tournament_hints_data.csv')

try:
    result_df = pd.read_csv('naics_codes.csv')
    start_index = result_df.index[-1]
    count = start_index
except:
    result_df = pd.DataFrame(columns=['name', 'naics_code', 'naics_label'])
    start_index = 0
    count = 0

df_sliced = df.loc[start_index:]
for index, row in tqdm(df_sliced.iterrows()):
    count += 1
    replacements = {
        'name': row['commercial_name'],
        'tags': row['business_tags'],
        'description': row['short_description'],
        'category': row['main_business_category']
    }
    prompt = prompt_template.format(**replacements)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    response = completion.choices[0].message.content
    if response.find('-') != -1:
        result_df = result_df._append({
            'name': row['commercial_name'],
            'naics_code': response.split(' - ')[0].strip(),
            'naics_label': response.split('-')[1].strip()
        }, ignore_index=True)
    else:
        result_df = result_df._append({
            'name': row['commercial_name'],
            'naics_code': response,
            'naics_label': ''
        }, ignore_index=True)
    if count % 10 == 0:
        result_df.to_csv('naics_codes.csv', index=False)

result_df.to_csv('naics_codes.csv', index=False)







