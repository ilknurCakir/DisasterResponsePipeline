import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer


def data_wrangling(df, n):
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    temp = df.iloc[:, 4:]
    a = pd.DataFrame({'Average': temp.mean()})
    a = a[a.index != 'child_alone']
    a = a.sort_values(by = 'Average', ascending = True)
    
    genre_count = pd.DataFrame({'genre_counts': df.genre.value_counts()})
    
    vect = CountVectorizer(stop_words = stopwords.words('english'), ngram_range = (1, 2))
    vected = vect.fit_transform(df.message)
    sum_words = vected.sum(axis = 0)
    words_freq = [[word, sum_words[0, idx]] for word, idx in vect.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_info = pd.DataFrame(words_freq[:n])
    
    return genre_names, genre_counts, genre_count, a, words_info
    
    