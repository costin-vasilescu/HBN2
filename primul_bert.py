pip install transformers

pip install accelerate -U

pip install datasets

pip install evaluate

import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from transformers import AutoTokenizer
import evaluate
from transformers import DataCollatorWithPadding
import numpy as np

from transformers import AutoModelForSequenceClassification


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


def mapping_labels(dataframe):

  dictionary_labels = {}


  for i in range(len(dataframe)):

    naics_code = dataframe['naics_code'][i]

    dictionary_labels[naics_code] = int(i)

  return dictionary_labels

df = pd.read_csv('label_taxonomy.csv')
dictionary = mapping_labels(df)

df_naics_codes = pd.read_csv('cleaned_naics_codes.csv')

df_naics_codes['label'] = df_naics_codes['naics_code'].map(dictionary)

df_naics_codes = df_naics_codes.dropna(subset = ['naics_code'])


df_naics_codes = df_naics_codes.dropna(subset = ['label'])
df_naics_codes['label'] = df_naics_codes['label'].astype(int)


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

dataset = Dataset.from_pandas(df_naics_codes)

dataset = dataset.train_test_split(test_size=0.2)

def preprocess_function(examples):

    return tokenizer(examples["name"], truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels = len(dictionary))

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()