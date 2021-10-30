import pandas as pd
import nltk
from collections import Counter
from itertools import chain
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
import time

nltk.download('stopwords')
start_time = time.time()
data = 'Corona_NLP_train.csv'
df = pd.read_csv(data)
print(df)
df.info()
df.describe()
print("\nUnique values :  \n", df.Sentiment.unique())  # shows all the classes that Sentiment can be
print(df.Sentiment.mode())
g = df.groupby('Sentiment').size().reset_index()  # lists all most common classifications
g = g.sort_values('Sentiment', ascending=False)    # sorts the classes by descending order
print(g)
print("\nShow the second most common sentiment :\n", g.iloc[1, :])  # shows the second most common classification
good_date = df.groupby(['TweetAt', 'Sentiment']).size().reset_index(name="Times")   # finds all the combinations of dates and classinficaions and how many times they appear
date = good_date[good_date['Sentiment'] == 'Extremely Positive']  # finds dates with the Extremely positive Sentiment
best_date = date.sort_values('Times', ascending=False)  # sorts the dates by frequency in descending order
print("\nList of best dates : \n ", best_date)
print("\nDate with the most positive posts : \n", best_date.head(1))
df['OriginalTweet'] = df['OriginalTweet'].str.lower()  # converts tweets into lower case
print("\n All the lower case messages :\n ", df['OriginalTweet'])
df['OriginalTweet'] = df['OriginalTweet'] = df.OriginalTweet.str.replace(r'[^a-zA-Z]+', r' ', regex=True)  # removes non-alphabetical characters
print("\n All the messages without non-alphabetic chars :\n ", df['OriginalTweet'])
df['OriginalTweet'] = df.OriginalTweet.str.replace('\s+', ' ', regex=True)   # removes unnecessary spaces
print("\n All the messages without unnecessary spaces :\n ", df['OriginalTweet'])
token = df['OriginalTweet'].str.split()  # splits the words
print("\n Tokenized words : \n", token)
word_list = list(chain.from_iterable(token.tolist()))  # converts a nested list into a single list
print("\n Number of tokenized words : \n", len(word_list))
unique = set(word_list)  # finds unique words in the list
print("\n Number of unique tokenized words : \n", len(unique))
c = Counter(word_list)  # finds word frequency in the list
print("\n The 10 most common words : \n", c.most_common(10))
stop_words = set(stopwords.words('english'))  # finds the unique stop words in the list to increase perfomances
filtered_words = [w for w in word_list if len(w) > 2]   # removes words with 2 or less characters
filtered_words = [w for w in filtered_words if w not in stop_words]  # removes stop words from the list
print("\n Number of filtered words : \n", len(filtered_words))
c = Counter(filtered_words)   # finds the word frequency in the filtered list
print("\n The 10 most common filtered words : \n", c.most_common(10))

for i in range(41157):   # filtering
    token[i] = [w for w in token[i] if len(w) > 2]  # removes words with 2 or less characters from all documents
    token[i] = [w for w in token[i] if w not in stop_words]  # removes stop words from the list from all documents

word_set_list = []
for i in range(41157):
    word_set_list.append(list(set(token[i])))  # makes a nested list of the unique words in each document

filtered_set_words = list(chain.from_iterable(word_set_list))  # coverts the list of list of unique words in each document into a single list
c_set = Counter(filtered_set_words)  # finds the word frequency over all document
print("\n The 10 Most frequent words over all documents :\n", c_set.most_common(10))
c_set = c_set.most_common(500)
dc_set = dict(c_set)
data_words = dc_set.keys()
words_counts = dc_set.values()
indexes = np.arange(len(data_words))
f = plt.figure()
plt.plot(indexes, words_counts)  # plots the word frequency over all documents
# plt.show()
f.savefig("plot.pdf", bbox_inches='tight')

df['OriginalTweet '] = word_set_list
print("\n word set list :\n", df['OriginalTweet'])
#y = df['Sentiment']
le = preprocessing.LabelEncoder()
#y = y.applle.fit_transform)
y = le.fit_transform(df['Sentiment'])
print("\n label set :\n", y)
x = df.drop('Sentiment', axis=1)
#x = np.array(x)
print("\n data set :\n", x)


print("--- %s seconds ---" % (time.time() - start_time))

