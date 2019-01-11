import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin



#------------------------------------- tokenize --------------------------------
def tokenize(text):
    '''Cleans, removes punctuation, tokenize, remove stop words and
    lemmatize
    
    Args:
        text (str): text data to be tokenized
    
    Returns:
        words (str): list of tokenized words
    '''
    
    # punctuation removal
    text = text.lower().replace(r'[^a-zA-Z0-9]', " ")
    words = word_tokenize(text) # tokenization
    
    #removing stop words and lemmatization
    words = [w for w in words if w not in stopwords.words('english')]
    words = [WordNetLemmatizer().lemmatize(w, pos = 'v') for w in words]
    
    return words


#----------------------------------- length extractor -----------------------
class LengthExtractor(BaseEstimator, TransformerMixin):
    '''Length Extractor transformer
    Attributes:
        
    
    '''
    def __int__(self):
        pass
    
    def func(self, text):
        '''Gives the number of sentences and the number of words in the text
        
        Args:
            text (str): lines of string to be get feature-extracted
            
        Returns:
            [sent_len, word_len] : list od integers representing the number of 
            sentences and the number of words
        '''
        sents = sent_tokenize(text)
        sent_len = len(sents)
        
        text = text.lower().replace(r'[^a-zA-Z0-9]', ' ')
        words = word_tokenize(text)
        word_len = len(words)
        
        return [sent_len, word_len]
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        '''Transforms data to dataframe of 2 columns giving the number
        of sentences of the number of words in the data
        
        Args:
            X (str): data to be feature-extracted
            
        Returns:
            df : dataframe of 2 columns
        '''
        
        X_len = pd.Series(X).apply(self.func)
        
        return pd.DataFrame(X_len.tolist())
    
    def fit_transform(self, X, y = None):
        '''fit and transform the data
        '''
        X = pd.Series(X).apply(self.func)
        
        return pd.DataFrame(X.tolist())
    
    
#---------------------------------- digit extractor ------------------------------------
class DigitExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def isDigit(self, text):
        '''Function to give if a text includes a digit or not
        
        Args:
            text (str): text data to be feature-extracted
            
        Returns:
            bool : Return value. True if successful, False otherwise
        '''
        
        tokens = word_tokenize(text)
        dgit = [t for t in tokens if t.isdigit()]
        
        if len(dgit) > 0:
            return True
        else:
            return False
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        '''Transforms data to dataframe of 1 column of dummy variable
        True if there is a digit, False if not.
        
        Args:
            X (str): data to be feature-extracted
            
        Returns:
            df : dataframe of 1 column with Boolean values
        '''
        
        X_digit = pd.Series(X).apply(self.isDigit)
        
        return pd.DataFrame(X_digit)
    
    def fit_transform(self, X, y = None):
        X_digit = pd.Series(X).apply(self.isDigit)
        return pd.DataFrame(X_digit)
 

#------------------------------------ url extractor ----------------------------------------
class UrlExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def is_url(self, text):
        '''Function to give if the text includes a url or not
        
        Args:
            text (str): text data to be feature-extracted
            
        Returns:
            bool : Return value. True if successful, False otherwise
        '''
        
        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        placer = re.findall(url_regex, text)
        
        if len(placer) > 0:
            return True
        else:
            return False
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        '''Transforms data to dataframe of 1 column of dummy variable
        True if there is a digit, False if not.
        
        Args:
            X (str): data to be feature-extracted
            
        Returns:
            df : dataframe of 1 column with Boolean values
        '''
                
        X_url = pd.Series(X).apply(self.is_url)
        
        return pd.DataFrame(X_url)
    
    def fit_transform(self, X, y = None):
        X = pd.Series(X).apply(self.is_url)
        
        return pd.DataFrame(X)
    
#-------------------------------------- gpe extractor --------------------------------------
class GpeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def hasgpe(self, text):
        '''Function to give if the text includes geopolitical, organization or
        person entities from nltk.ne_chunk 
        
        Args:
            text (str): text data to be feature-extracted
            
        Returns:
            bool : True if text includes desired entities, False otherwise
        '''

        sentences = sent_tokenize(text)
        result = False
        for sent in sentences:
            tokens = word_tokenize(sent)
            tags = pos_tag(tokens)
            chunks = ne_chunk(tags)
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    if chunk.label() in ['GPE', 'Organization']:
                        result = True
                        return True
                        #break
 
        return result
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        '''Transforms data to dataframe of 1 column of dummy variable
        True if there is GPE, Location or Person entities from nltk.ne_chunk
        
        Args:
            X (str): data to be feature-extracted
            
        Returns:
            df : dataframe of 1 column with Boolean values
        '''
        
        X_verb = pd.Series(X).apply(self.hasgpe)
        return pd.DataFrame(X_verb)
    
    def fit_transform(self, X, y = None):
        X = pd.Series(X).apply(self.hasgpe)
        
        return pd.DataFrame(X)
    