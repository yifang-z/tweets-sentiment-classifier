import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
import pickle
warnings.filterwarnings('ignore')
df = pd.read_csv('twitter-training-data.txt', delimiter='\t', header=None)#首先读成df
batch_1 = df[:100]
batch_1[1].value_counts()
label2id = {
    'positive': -1,
    'neutral': 0,
    'negative': 1,
}
batch_1[1] = [label2id[label_str] for label_str in batch_1[1]]

df = pd.read_csv('twitter-test1.txt', delimiter='\t', header=None)#首先读成df
batch_2 = df
batch_2[1].value_counts()
label2id = {
    'positive': -1,
    'neutral': 0,
    'negative': 1,
}
batch_2[1] = [label2id[label_str] for label_str in batch_2[1]]

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
tokenized_1 = batch_1[2].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
max_len = 0
for i in tokenized_1.values:
    if len(i) > max_len:
        max_len = len(i)
padded_1 = np.array([i + [0]*(max_len-len(i)) for i in tokenized_1.values])
np.array(padded_1).shape
attention_mask_1 = np.where(padded_1 != 0, 1, 0)
attention_mask_1.shape
input_ids_1 = torch.tensor(padded_1).to(torch.int64)
attention_mask_1 = torch.tensor(attention_mask_1).to(torch.int64)

tokenized_2 = batch_2[2].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
max_len = 0
for i in tokenized_2.values:
    if len(i) > max_len:
        max_len = len(i)
padded_2 = np.array([i + [0]*(max_len-len(i)) for i in tokenized_2.values])
np.array(padded_2).shape
attention_mask_2 = np.where(padded_2 != 0, 1, 0)
attention_mask_2.shape
input_ids_2 = torch.tensor(padded_2).to(torch.int64)
attention_mask_2 = torch.tensor(attention_mask_2).to(torch.int64)
with torch.no_grad():
    last_hidden_states_1 = model(input_ids_1, attention_mask=attention_mask_1)
    last_hidden_states_2 = model(input_ids_2, attention_mask=attention_mask_2)
features_1 = last_hidden_states_1[0][:,0,:].numpy()
labels_1 = batch_1[1]
features_2 = last_hidden_states_2[0][:,0,:].numpy()
labels_2 = batch_2[1]
bert = LogisticRegression()
bert.fit(features_1, labels_1)
bert.score(features_2, labels_2)
s = pickle.dumps(bert)
with open('bert.model','wb+') as f:#注意此处mode是'wb+'，表示二进制写入
    f.write(s)
