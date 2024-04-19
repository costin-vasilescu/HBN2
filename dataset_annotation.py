from openai import OpenAI
import os
import pandas as pd

from dotenv import load_dotenv
load_dotenv()


# Load the dataset
# df = pd.read_csv('tournament_hints_data.csv')

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Write a very short poem"},
  ]
)
print(completion.choices[0].message)