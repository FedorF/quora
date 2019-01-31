from source.models import FeedForward, LSTM
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from torchtext import data, vocab
from tqdm import tqdm
from nltk.corpus import stopwords
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from source.utils import train_model
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

PATH_TO_MODEL = './models/lstm'
PATH_TO_EMB_FOLDER = './data/glove.840B.300d'
PATH_TO_EMB_FILE = './data/glove.840B.300d/glove.840B.300d.txt'
PATH_TO_TRAINING_DATA = './data/train.csv'
PATH_TO_TEST_DATA = './data/test.csv'
PATH_TO_SMALL_DATA = './data/train_small.csv'
BATCH_SIZE = 128
learning_rate = 3e-4
num_epoch = 3

nlp = spacy.load('en', disable=['parser', 'ner'])


# stop_words = stopwords.words('english')

def tokenizer(text):
    return [token.text for token in nlp.tokenizer(text)]

qid = None
text_field = data.Field(sequential=True, tokenize=tokenizer, lower=True)
target_field = data.Field(sequential=False, use_vocab=False, is_target=True, dtype=torch.long)

df = data.TabularDataset(path=PATH_TO_TRAINING_DATA,
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
text_field.build_vocab(df, df_test, vectors=vec)


train, val = df.split(split_ratio=[0.8, 0.2])
train_dl, val_dl = data.Iterator.splits((train, val),
                                        sort_key=lambda t: len(t.question_text),
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

model = LSTM(text_field.vocab.vectors, num_layers=2)

optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()

loss_hist, train_acc_hist, val_acc_hist = train_model(model, optim,
                                                       loss, num_epoch,
                                                       train_dl, val_dl,
                                                       BATCH_SIZE,
                                                       PATH_TO_MODEL,
                                                       verbose_every=1000)

plt.plot(train_acc_hist)
plt.plot(val_acc_hist)


val_pred = []
val_true = []

for batch in tqdm(val_dl):
    output = model(batch.question_text)
    val_pred += nn.functional.sigmoid(output.detach()).numpy()[:, 1].tolist()
    val_true += batch.target.numpy().tolist()

tmp = [0, 0, 0] # idx, cur, max
delta = 0
for tmp[0] in np.arange(0.1, 0.501, 0.01):
    tmp[1] = f1_score(val_true, val_pred > tmp[0])
    if tmp[1] > tmp[2]:
        delta = tmp[0]
        tmp[2] = tmp[1]
print('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, tmp[2]))