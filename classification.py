from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from naive_bayes import NaiveBayes
from LogisticRegressionRe import LogisticRegressionRe
import numpy as np
from MaxEntropy import maxEnt
import pickle
from LSTM import lstm_m

# Global Parameters
stop_words = set(stopwords.words('english'))
# # f = open("twitter-training-data.txt", 'r', encoding="utf-8")
# # i = 0
# # preProList = []
# # bow = []
# # lable = list(())
# # tList = []
# # for line in f.readlines():
# #     tweets_vectorized_train = []
# #     tweet = line.split('\t')[2]
# #     if line.split('\t')[1] == "positive":
# #         lable.append(1)
# #     if line.split('\t')[1] == "negative":
# #         lable.append(2)
# #     if line.split('\t')[1] == "neutral":
# #         lable.append(0)
# #     tweet.lower()
# #     # Remove urls
# #     tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
# #     # Remove user @ references and '#' from tweet
# #     tweet = re.sub(r'\@\w+|\#', '', tweet)
# #     # Remove punctuations
# #     tweet = tweet.translate(str.maketrans('', '', string.punctuation))
# #     # Remove stopwords
# #     tweet_tokens = word_tokenize(tweet)
# #     filtered_words = [w for w in tweet_tokens if not w in stop_words]
# #     tList.append(' '.join(filtered_words))
# # bow_vectorizer = TfidfVectorizer (max_features=1000, stop_words='english')
# # bow = bow_vectorizer.fit_transform(tList).toarray()
# #
# # f = open("twitter-test1.txt", 'r', encoding="utf-8")
# # i = 0
# # preProList = []
# # bow_test = []
# # teList = []
# # lable_t = list(())
# # for line in f.readlines():
# #     tweets_vectorized_train = []
# #     tweet = line.split('\t')[2]
# #     if line.split('\t')[1] == "positive":
# #         lable_t.append(1)
# #     if line.split('\t')[1] == "negative":
# #         lable_t.append(2)
# #     if line.split('\t')[1] == "neutral":
# #         lable_t.append(0)
# #     tweet.lower()
# #     # Remove urls
# #     tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
# #     # Remove user @ references and '#' from tweet
# #     tweet = re.sub(r'\@\w+|\#', '', tweet)
# #     # Remove punctuations
# #     tweet = tweet.translate(str.maketrans('', '', string.punctuation))
# #     # Remove stopwords
# #     tweet_tokens = word_tokenize(tweet)
# #     filtered_words = [w for w in tweet_tokens if not w in stop_words]
# #     teList.append(' '.join(filtered_words))
# # bow_test = bow_vectorizer.transform(teList).toarray()
# #
# #
# # f = open("twitter-test2.txt", 'r', encoding="utf-8")
# # i = 0
# # preProList = []
# # bow_test2 = []
# # teList = []
# # lable_t2 = list(())
# # for line in f.readlines():
# #     tweets_vectorized_train = []
# #     tweet = line.split('\t')[2]
# #     if line.split('\t')[1] == "positive":
# #         lable_t2.append(1)
# #     if line.split('\t')[1] == "negative":
# #         lable_t2.append(2)
# #     if line.split('\t')[1] == "neutral":
# #         lable_t2.append(0)
# #     tweet.lower()
# #     # Remove urls
# #     tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
# #     # Remove user @ references and '#' from tweet
# #     tweet = re.sub(r'\@\w+|\#', '', tweet)
# #     # Remove punctuations
# #     tweet = tweet.translate(str.maketrans('', '', string.punctuation))
# #     # Remove stopwords
# #     tweet_tokens = word_tokenize(tweet)
# #     filtered_words = [w for w in tweet_tokens if not w in stop_words]
# #     teList.append(' '.join(filtered_words))
# # bow_test2 = bow_vectorizer.transform(teList).toarray()
# #
# # f = open("twitter-test3.txt", 'r', encoding="utf-8")
# # i = 0
# # preProList = []
# # bow_test3 = []
# # teList = []
# # lable_t3 = list(())
# # for line in f.readlines():
# #     tweets_vectorized_train = []
# #     tweet = line.split('\t')[2]
# #     if line.split('\t')[1] == "positive":
# #         lable_t3.append(1)
# #     if line.split('\t')[1] == "negative":
# #         lable_t3.append(2)
# #     if line.split('\t')[1] == "neutral":
# #         lable_t3.append(0)
# #     tweet.lower()
# #     # Remove urls
# #     tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
# #     # Remove user @ references and '#' from tweet
# #     tweet = re.sub(r'\@\w+|\#', '', tweet)
# #     # Remove punctuations
# #     tweet = tweet.translate(str.maketrans('', '', string.punctuation))
# #     # Remove stopwords
# #     tweet_tokens = word_tokenize(tweet)
# #     filtered_words = [w for w in tweet_tokens if not w in stop_words]
# #     teList.append(' '.join(filtered_words))
# # bow_test3 = bow_vectorizer.transform(teList).toarray()
# #
# # NB = NaiveBayes()
# # NB.fit(bow, lable)
# # y_pred = NB.predict(bow_test)
# # y_pred2 = NB.predict(bow_test2)
# # y_pred3 = NB.predict(bow_test3)
# # print(accuracy_score(lable_t, y_pred))
# # print(accuracy_score(lable_t2, y_pred2))
# # print(accuracy_score(lable_t3, y_pred3))
#
# stop_words = set(stopwords.words('english'))
# f = open("twitter-training-data.txt", 'r', encoding="utf-8")
# i = 0
# preProList = []
# bow = []
# label = list(())
# labelText = list(())
# tList = []
# for line in f.readlines():
#     tweets_vectorized_train = []
#     tweet = line.split('\t')[2]
#     if line.split('\t')[1] == "positive":
#         label.append(1)
#     if line.split('\t')[1] == "negative":
#         label.append(2)
#     if line.split('\t')[1] == "neutral":
#         label.append(0)
#     labelText.append(line.split('\t')[1])
#     tweet.lower()
#     # Remove urls
#     tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
#     # Remove user @ references and '#' from tweet
#     tweet = re.sub(r'\@\w+|\#', '', tweet)
#     # Remove punctuations
#     tweet = tweet.translate(str.maketrans('', '', string.punctuation))
#     # Remove stopwords
#     tweet_tokens = word_tokenize(tweet)
#     filtered_words = [w for w in tweet_tokens if not w in stop_words]
#     tList.append(' '.join(filtered_words))
#
# f = open("twitter-test1.txt", 'r', encoding="utf-8")
# i = 0
# preProList = []
# bow_test = []
# teList = []
# label_t = list(())
# labelText_t = list(())
# for line in f.readlines():
#     tweets_vectorized_train = []
#     tweet = line.split('\t')[2]
#     if line.split('\t')[1] == "positive":
#         label_t.append(1)
#     if line.split('\t')[1] == "negative":
#         label_t.append(2)
#     if line.split('\t')[1] == "neutral":
#         label_t.append(0)
#     labelText_t.append(line.split('\t')[1])
#     tweet.lower()
#     # Remove urls
#     tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
#     # Remove user @ references and '#' from tweet
#     tweet = re.sub(r'\@\w+|\#', '', tweet)
#     # Remove punctuations
#     tweet = tweet.translate(str.maketrans('', '', string.punctuation))
#     # Remove stopwords
#     tweet_tokens = word_tokenize(tweet)
#     filtered_words = [w for w in tweet_tokens if not w in stop_words]
#     teList.append(' '.join(filtered_words))
#
# f = open("twitter-test2.txt", 'r', encoding="utf-8")
# i = 0
# preProList = []
# bow_test2 = []
# teList = []
# label_t2 = list(())
# labelText_t2 = list(())
# for line in f.readlines():
#     tweets_vectorized_train = []
#     tweet = line.split('\t')[2]
#     if line.split('\t')[1] == "positive":
#         label_t2.append(1)
#     if line.split('\t')[1] == "negative":
#         label_t2.append(2)
#     if line.split('\t')[1] == "neutral":
#         label_t2.append(0)
#     labelText_t2.append(line.split('\t')[1])
#     tweet.lower()
#     # Remove urls
#     tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
#     # Remove user @ references and '#' from tweet
#     tweet = re.sub(r'\@\w+|\#', '', tweet)
#     # Remove punctuations
#     tweet = tweet.translate(str.maketrans('', '', string.punctuation))
#     # Remove stopwords
#     tweet_tokens = word_tokenize(tweet)
#     filtered_words = [w for w in tweet_tokens if not w in stop_words]
#     teList.append(' '.join(filtered_words))
#
# f = open("twitter-test3.txt", 'r', encoding="utf-8")
# i = 0
# preProList = []
# bow_test3 = []
# teList = []
# label_t3 = list(())
# labelText_t3 = list(())
# for line in f.readlines():
#     tweets_vectorized_train = []
#     tweet = line.split('\t')[2]
#     if line.split('\t')[1] == "positive":
#         label_t3.append(1)
#     if line.split('\t')[1] == "negative":
#         label_t3.append(2)
#     if line.split('\t')[1] == "neutral":
#         label_t3.append(0)
#     labelText_t3.append(line.split('\t')[1])
#     tweet.lower()
#     # Remove urls
#     tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
#     # Remove user @ references and '#' from tweet
#     tweet = re.sub(r'\@\w+|\#', '', tweet)
#     # Remove punctuations
#     tweet = tweet.translate(str.maketrans('', '', string.punctuation))
#     # Remove stopwords
#     tweet_tokens = word_tokenize(tweet)
#     filtered_words = [w for w in tweet_tokens if not w in stop_words]
#     teList.append(' '.join(filtered_words))

# NB = MultinomialNB()
# NB.fit(bow, lable)
# y_pred = NB.predict(bow_test)
# y_pred2 = NB.predict(bow_test2)
# y_pred3 = NB.predict(bow_test3)
# print(accuracy_score(lable_t, y_pred))
# print(accuracy_score(lable_t2, y_pred2))
# print(accuracy_score(lable_t3, y_pred3))

# lstm_model = LSTM_model(tList, teList, labelText, labelText_t3)
# lstm_model.read_GloVe_data("glove.6B.100d.txt", 100)
# lstm_model.train_model()

# LR_model = LogisticRegression()
# LR_model.fit(bow, lable)
# y_predict_lr = LR_model.predict(bow_test)
# y_predict_lr2 = LR_model.predict(bow_test2)
# y_predict_lr3 = LR_model.predict(bow_test3)
# print(accuracy_score(lable_t, y_predict_lr))
# print(accuracy_score(lable_t2, y_predict_lr2))
# print(accuracy_score(lable_t3, y_predict_lr3))
# #
def dataclean(filename):
    f = open(filename, 'r', encoding="utf-8")
    i = 0
    tList = []
    label = list(())
    labelText = list(())
    for line in f.readlines():
        tweets_vectorized_train = []
        tweet = line.split('\t')[2]
        labelText.append(line.split('\t')[1])
        if line.split('\t')[1] == "positive":
            label.append(1)
        if line.split('\t')[1] == "negative":
            label.append(2)
        if line.split('\t')[1] == "neutral":
            label.append(0)
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
    return tList, labelText, label

# tList, labelText, label = dataclean("twitter-training-data.txt")
# bow = []
# bow_test = []
# bow_vectorizer = TfidfVectorizer (max_features=1000, stop_words='english')
# bow = bow_vectorizer.fit_transform(tList).toarray()

# teList, labeleText, label_t = dataclean("twitter-test2.txt")
# bow_test = bow_vectorizer.transform(teList).toarray()
#
# LR_model = LogisticRegressionRe()
# LR_model.fit(bow, label)
# y_predict_lr = LR_model.predict(bow_test)
# print(accuracy_score(label_t, y_predict_lr))
# s = pickle.dumps(LR_model)
# with open('logisticRe.model','wb+') as f:#注意此处mode是'wb+'，表示二进制写入
#     f.write(s)
# f = open('logisticRe.model','rb') #注意此处model是rb
# s = f.read()

# logistics_model = pickle.loads(s)
# y_pred = logistics_model.predict(bow_test)
# lable_t = np.array(lable_t)
# y_pred = np.reshape(y_pred, lable_t.shape)
# print(accuracy_score(lable_t, y_pred))
# y_pred = logistics_model.predict(bow_test2)
# lable_t2 = np.array(lable_t2)
# y_pred = np.reshape(y_pred, lable_t2.shape)
# print(accuracy_score(lable_t2, y_pred))
# y_pred = logistics_model.predict(bow_test3)
# lable_t3 = np.array(lable_t3)
# y_pred = np.reshape(y_pred, lable_t3.shape)
# print(accuracy_score(lable_t3, y_pred))

# maxEnt = maxEnt(bow, lable)
# maxEnt.maxEntropyTrain()
# y_pred = maxEnt.test(bow_test)
# print(accuracy_score(lable_t, y_pred))
# s = pickle.dumps(maxEnt)
# with open('myModel.model','wb+') as f:#注意此处mode是'wb+'，表示二进制写入
#     f.write(s)
# f = open('ME.model','rb') #注意此处model是rb
# s = f.read()
# model = pickle.loads(s)
# y_pred = model.test(bow_test)
# print(accuracy_score(lable_t, y_pred))
# y_pred = model.test(bow_test2)
# print(accuracy_score(lable_t2, y_pred))
# y_pred = model.test(bow_test3)
# print(accuracy_score(lable_t3, y_pred))
lstm_model = lstm_m()
lstm_model.train_lstm("twitter-test1.txt")
s = pickle.dumps(lstm_model)
with open('lstm.model','wb+') as f:#注意此处mode是'wb+'，表示二进制写入
    f.write(s)
f = open('lstm.model','rb') #注意此处model是rb
s = f.read()
model = pickle.loads(s)
model.train_lstm("twitter-test2.txt")