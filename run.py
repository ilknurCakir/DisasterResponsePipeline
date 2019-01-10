import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import random

from flask import Flask
import plotly.graph_objs as go
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from utils import LengthExtractor, DigitExtractor, UrlExtractor, GpeExtractor
from data_wrangling import data_wrangling

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
path_to_clean = 'sqlite:///' + 'data/DisasterResponse.db'
engine = create_engine(path_to_clean)
#engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('clean', con = engine)
#print(df.head(4))

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    n = 150
    genre_names, genre_counts, genre_count, a, words_info = data_wrangling(df, n)


    random.seed(37)
    colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(n-1)]
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, {
            'data': [
                Bar(
                    x=a.Average,
                    y=a.index,
                    orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Average Values of Categories',
                'height': 700,
                'width': 1000,
                'margin': 200
            }
        }, {
            "data": [{
                "values": genre_count.genre_counts.tolist(),
                "labels": [
                    "news",
                    "direct",
                    "social"],
                "domain": {"x": [0, 1]},
                "hoverinfo":"label+percent",
                "hole": .4,
                "type": "pie"
    }],

            "layout": {
                "title":"Genre Distribution"
                }
    }, {
            'data': [{
                    'x' : [random.random() for i in range(n-1)],
                    'y' : [random.random() for i in range(n-1)],
                    'mode': 'text',
                    'text' : words_info[0],
                    'marker' : {'opacity': 0.3},
                    'textfont': {'size': words_info[1]/27,
                           'color': colors}
            }],

            'layout': {
                'title': 'Word Cloud of the Most Frequent 150 Words <br> with Text Size Icreasing with Frequency',
                'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                'height' : 600,
                'width' : 1200
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master2.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
    
# Reference for Word Cloud : https://community.plot.ly/t/wordcloud-in-dash/11407/4