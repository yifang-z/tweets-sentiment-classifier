from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import numpy as np
import pickle
import nltk
import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import load_model

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


class lstm_m:
    def __init__(self):
        self.a = []
    def read_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            word_vocab = set()
            word2vector = {}
            for line in f:
                line_ = line.strip()
                words_Vec = line_.split()
                word_vocab.add(words_Vec[0])
                word2vector[words_Vec[0]] = np.array(words_Vec[1:], dtype=float)
        return word_vocab, word2vector
    def train_lstm(self,textfilename):
        f = open("twitter-training-data.txt", 'r', encoding="utf-8")
        i = 0
        lable = list(())
        tList = []
        for line in f.readlines():
            tweets_vectorized_train = []
            tweet = line.split('\t')[2]
            lable.append(line.split('\t')[1])
            tweet.lower()
            # Remove urls
            tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
            # Remove user @ references and '#' from tweet
            tweet = re.sub(r'\@\w+|\#', '', tweet)
            # Remove punctuations
            tweet = tweet.translate(str.maketrans('', '', string.punctuation))
            # Remove stopwords
            tweet_tokens = word_tokenize(tweet)
            filtered_words = [w for w in tweet_tokens if not w in stop_words]
            tList.append(' '.join(filtered_words))

        f = open(textfilename, 'r', encoding="utf-8")
        i = 0
        teList = []
        lable_t = list(())
        for line in f.readlines():
            tweets_vectorized_train = []
            tweet = line.split('\t')[2]
            lable_t.append(line.split('\t')[1])
            tweet.lower()
            # Remove urls
            tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
            # Remove user @ references and '#' from tweet
            tweet = re.sub(r'\@\w+|\#', '', tweet)
            # Remove punctuations
            tweet = tweet.translate(str.maketrans('', '', string.punctuation))
            # Remove stopwords
            tweet_tokens = word_tokenize(tweet)
            filtered_words = [w for w in tweet_tokens if not w in stop_words]
            teList.append(' '.join(filtered_words))

        tokenizer = Tokenizer(num_words=None, split=' ')
        tokenizer.fit_on_texts(tList + teList)
        train_text_vec = tokenizer.texts_to_sequences(tList)
        test_text_vec = tokenizer.texts_to_sequences(teList)
        train_text_vec = pad_sequences(train_text_vec, maxlen=50)
        test_text_vec = pad_sequences(test_text_vec, maxlen=50)


        w_idx = tokenizer.word_index
        vocab, word_to_idx = self.read_data("glove.6B.100d.txt")
        embedding_matrix = np.zeros((len(w_idx) + 1, 100))
        for word, i in w_idx.items():
            embedding_vector = word_to_idx.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        encoder = LabelEncoder()
        lable = encoder.fit_transform(lable)
        lable = to_categorical(lable)
        lable_t = encoder.fit_transform(lable_t)
        lable_t = to_categorical(lable_t)

        model = Sequential()
        model.add(
            Embedding(len(w_idx) + 1, 100, input_length=train_text_vec.shape[1], weights=[embedding_matrix],
                      trainable=False))
        model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))
        model.add(keras.layers.core.Dense(3, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        model.fit(train_text_vec, lable, epochs=1, batch_size=32)
        print(model.evaluate(test_text_vec, lable_t)[1])

#

