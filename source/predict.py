import torch
import torch.nn as nn
import pandas as pd
from torchtext import data, vocab
import spacy
from source.models import FeedForward, LSTM
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import pandas as pd


PATH_TO_MODEL = './models/lstm'
PATH_TO_EMB_FOLDER = './data/glove.840B.300d'
PATH_TO_EMB_FILE = './data/glove.840B.300d/glove.840B.300d.txt'
PATH_TO_TEST_DATA = './data/test.csv'
PATH_TO_TRAINING_DATA = './data/train.csv'
PATH_TO_SUB = './sub.csv'

BATCH_SIZE = 1

nlp = spacy.load('en', disable=['parser', 'ner'])

def tokenizer(text):
    return([token.text for token in nlp.tokenizer(text)])



qid = data.Field(sequential=False, use_vocab=False, dtype=torch.long)
text_field = data.Field(sequential=True, tokenize=tokenizer, lower=True)
target_field = None

df_train = data.TabularDataset(path=PATH_TO_TRAINING_DATA,
                         format='CSV',
                         fields=[('qid', qid),
                                 ('question_text', text_field),
                                 ('target', target_field)],
                         skip_header=True)

df_test = data.TabularDataset(path=PATH_TO_TEST_DATA,
                              format='CSV',
                              fields=[('qid', qid),
                                      ('question_text', text_field)],
                              skip_header=True)

vec = vocab.Vectors(PATH_TO_EMB_FILE)
text_field.build_vocab(df_train, df_test, vectors=vec)

test = data.Iterator(df_test,
                     batch_size=BATCH_SIZE,
                     sort_key=lambda t: len(t.question_text))

model = LSTM(text_field.vocab.vectors, num_layers=2)
model.load_state_dict(torch.load(PATH_TO_MODEL))
model.eval()

submission = {}
for batch in tqdm(test):
    output = model(batch.question_text)
    if nn.functional.sigmoid(output.detach()).numpy()[:, 1].tolist()[0] > 0.5:
        output = 1
    else:
        output = 0
    submission[batch.qid.numpy()[0]] = output

result = pd.DataFrame.from_dict(submission, orient='index')
result.to_csv(PATH_TO_SUB)