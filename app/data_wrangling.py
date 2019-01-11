import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer


def data_wrangling(df, n):
    ''' Takes df and returns data ready to create plotly
    graphs. 
    INPUTs:
    df - (dataframe) cleaned data including messages and categories
    n - (int) the number of the most frequent words in messages to
            be depicted in word cloud
            
    OUTPUTs:
    genre_names - (list) a list containing genre names
    genre_counts - (list) a list containing the counts of 
                    different genres
    genre_count - (dataframe) dataframe with columns of genres and 
                    their counts
    a - (dataframe) df with columns of different categories and
                    their average values. It gives the percentage
                    of messages included in these categories.
    words_info - (dataframe) the most frequent n words and their
                    frequencies
                    
    '''
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
    
    