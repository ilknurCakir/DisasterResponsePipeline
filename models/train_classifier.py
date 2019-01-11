#importing libraries
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
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, ne_chunk

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score, make_scorer, fbeta_score
from sklearn.metrics import precision_score, recall_score
import argparse
import pickle
from utils import tokenize, LengthExtractor, DigitExtractor, UrlExtractor, GpeExtractor

# getting inputs from keyboard
parser = argparse.ArgumentParser(description = 'Get database path and pickle file name')

parser.add_argument('dbase_path', help = 'Database filepath')
parser.add_argument('pickle_file', help = 'Pickle file name')

args = parser.parse_args()

database_filepath = args.dbase_path
model_filepath = args.pickle_file

######################################## loading data ###############################
def load_data(database_path):
    '''
    Load data from database and get it ready as X and y for
    scikit-learn.
    
    input args: None
    output args: X, y
    
    '''
    path = 'sqlite:///' + database_path

    # make a connection to the database and load data from database
    engine = create_engine(path)
    df = pd.read_sql_table('clean', con = engine)
    
    #preprocessing data
    #dropping 'child_alone' column as it includes only '0'
    df.drop('child_alone', axis = 1, inplace = True)
    
    # converting '2' to '1' in 'related' column
    df.loc[df.related == 2, 'related'] = 1
    
    # assigning 'message' column to X, and last 35 columns to y
    X = df.iloc[:, 1]
    y = df.iloc[:, 4:]
    colnames = y.columns.tolist()
    
    # dropping null values in X and y
    index = y[y.related.isnull()].index
    y.drop(index, axis= 0, inplace = True)
    X.drop(index, axis= 0, inplace = True)
    
    #converting dataframes to numpy arrays
    X = X.values
    y = y.values
    
    return X, y, colnames

#------------------------------- make pipeline ---------------------------------------
def make_pipeline(clf = None):
    '''
    Makes a pipeline 
    Length Extractor, DigitExtractor, UrlExtractor, GpeExtractor 
    and bow_pipeline extracts features in parallel, resulting features
    are concetanated. All are fed into a classifier; MultiOutputClassifier
    
    Input Args: None
    Output: returns a pipeline
    '''
    # make bag-of-words (bow) pipeline
    estimators = []
    estimators.append(('vect', CountVectorizer(ngram_range = (1, 2))))
    estimators.append(('tfidf', TfidfTransformer(sublinear_tf = True)))
    bow_pipeline = Pipeline(estimators)
    
    # makes FeatureUnion 
    features = []
    features.append(('length_ext', LengthExtractor()))
    features.append(('url_ext', UrlExtractor()))
    features.append(('digit_ext', DigitExtractor()))
    features.append(('gpe_ext', GpeExtractor()))
    features.append(('bow_pipeline', bow_pipeline))
    
    feature_union = FeatureUnion(features)
    
    #make classifier
    if  clf != None:        
        multi_clf = MultiOutputClassifier(clf) 
    else:
        multi_clf = MultiOutputClassifier(LogisticRegression(random_state = 42, class_weight = 'balanced'))
            
    # features --> classifier pipeline
    estimators = []
    estimators.append(('features', feature_union)) 
    estimators.append(('clf', multi_clf))
    pipeline = Pipeline(estimators)
        
    return pipeline 
#--------------------------------------make grid object -----------------------------------
def make_gridobject():
    
    parameters = {#'features__bow_pipeline__tfidf__ngram_range' : [(1, 1), (1, 2)],
                 'features__bow_pipeline__tfidf__sublinear_tf' : [True, False],
                 'features__bow_pipeline__vect__stop_words' : [None, stopwords.words('english')],
                 'features__bow_pipeline__vect__ngram_range' : [(1, 1), (1, 2)]} 
    
    pipeline = make_pipeline()
    
    scorer = make_scorer(accuracy_score)
    
    gridobj = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer, cv = 3)
    
    return gridobj
    
#------------------------------------ display results  ---------------------------------
def display_results(y_test, y_pred, colnames, cv = None):
    '''
    Displays the classification report for each column in y_test
    
    Input Args:
    cv : grid search object trained on training data
    y_test : Actual test values
    y_pred: Predicted test values
    
    Output Args: None
    '''
    
    # classification report for each column
    
    print('Results:')
    print('--------------------------------------------------------------')
    for col in range(y_test.shape[1]):
        print('{}'.format(colnames[col]))
        acc = accuracy_score(y_test[:, col], y_pred[:, col])
        pre = precision_score(y_test[:, col], y_pred[:, col])
        recall = recall_score(y_test[:, col], y_pred[:, col])
        fscore = fbeta_score(y_test[:, col], y_pred[:, col], beta = 1)
        print('Accuracy: {:.3f}      Precision: {:.3f}      Recall: {:.3f}      F-Score: {:.3f}\n'.format(acc, pre, recall, fscore))
        #print(classification_report(y_test[:, col], y_pred[:, col]), '\n')
        
    if cv != None:
    
        # Display best parameters of search object        
        print('Best Parameters:----------------------------------------------')
        print(cv.best_params_)
        
# -----------------------------compare different models-----------------------------------
def compare_fscore(models, y_preds, y_test):
    '''Compares f scores of different models for each category
    INPUTs:
    models - (list) list of models' names
    y_preds - (list) list of predicted values of different
                models to be compared
    y_test - (dataframe) df of test values to be used in 
                calculation of fscores
                
    OUTPUT:
    arr - (dataframe) dataframe with columns of fscores for each
            category when a model in models list is used in 
            MultiOutputClassifier
    '''
    
    arr = pd.DataFrame(0, index = [i for i in range(y_test.shape[1])], columns = models)
    for i in range(len(y_preds)):
        y_pred = y_preds[i]
        for col in range(y_test.shape[1]):
            
            arr.iloc[col, i] = round(fbeta_score(y_test[:, col], y_pred[:, col], beta = 1), 4)
            
    
    return arr


def compare_models():
    '''Compares the performance of different models as an input
    to MultiOutputClassifier
    OUTPUT:
    comparison_df - (dataframe) df with columns of f-score obtained 
                from each model
    '''
    
    
    clf_lst = [LogisticRegression(random_state = 42, class_weight = 'balanced'),
              MultinomialNB(),
              SVC(class_weight = 'balanced', kernel = 'rbf'),
              RandomForestClassifier(random_state = 42, max_depth = 3)]

    y_preds = []
    model_names = []

    for clf in clf_lst:
        model = make_pipeline(clf)
        model.fit(X_train, y_train)
        name = clf.__class__.__name__
        model_names.append(name)
        y_predicted= model.predict(X_test)
        y_preds.append(y_predicted)

    
    comparison_df = compare_fscore(model_names, y_preds, y_test)
    
    return comparison_df

#------------------------ saving the model ----------------------------------------------
def save_model(model, model_filepath):
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    
#-------------------------------- main -------------------------------------------
def main(cv_search = False):
    
    print('Loading data... \n   DATABASE: {}'.format(database_filepath))
    
    X, y, colnames = load_data(database_filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)
    
    if cv_search:
        
        print('Building model...')
        cv = make_gridobject()
        
        print('Training Model...')
        print('Making Grid Search for optimal parameters...')
        cv.fit(X_train, y_train)
        model = cv.best_estimator_
        
        print('Grid Search is done...')
        #print('Comparing different machine learning models...')
        #print(compare_models())
        
    else:
        print('Building model...')
        model = make_pipeline(LogisticRegression(random_state = 42, class_weight = 'balanced'))
        
        print('Training Model...')
        model.fit(X_train, y_train)
        
        print('Training is done...')
        
    print('Predicting test data...')
    pred = model.predict(X_test)
        
    print('Evaluating Model...')
    display_results(y_test, pred, colnames)
    
    print('Saving Model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)
    
    print('Trained model saved!')
    
    
    
if __name__ == "__main__":
    
    main()
    
    
    
    
