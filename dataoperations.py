import numpy as np 
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

df = pd.read_csv('Musical_instruments_reviews.csv', engine='python')

df.reviewText.fillna("",inplace = True)
#Deleting columns that are not informative
del df['reviewerID']
del df['asin']
del df['reviewerName']
del df['helpful']
del df['unixReviewTime']
del df['reviewTime']

df['text'] = df['reviewText'] + ' ' + df['summary']
del df['reviewText']
del df['summary']

def sentiment_rating(rating): #Label binarizing operations
    if(rating ==  1 or rating ==2 or rating == 3 ):
        return 0
    else:
        return 1
    
df.overall = df.overall.apply(sentiment_rating) 

stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
    
#Lemmatizing text data
def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            pos = pos_tag([i.strip()])
            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
            final_text.append(word.lower())
    return " ".join(final_text)

df.text= df.text.apply(lemmatize_words)

#print(df.overall.value_counts())

X = df.text
y = df.overall

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2 , random_state = 0)

#Converting words' values to 1s and 0s with Count Vectorizer
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
cv.fit(X_train)
cv_text=cv.transform(X_train)
cv_text_test = cv.transform(X_test)

#Count Vectorizer is saved to be used in mlmodels module later
joblib.dump(cv,'countvectorizer.pkl')

