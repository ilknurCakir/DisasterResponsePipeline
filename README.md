# DisasterResponse


Classification of messages sent/posted on social media during disaster events.
The data is from Figure Eight to build a model for an API that classifies 
messages.

How to Run Code:

models folder includes 2 files; process_data.py and train_classifier.py

process_data.py takes three inputs from keyboard; the filepath to the disaster
messages, the filepath to the categories messages and filepath to the database
to save the cleaned data. 

Python Script Ex: 

python models/process_data.py data/messages.csv data/categories.csv 
data/DisasterResponse.db

train_classifier.py takes 2 inputs; the filepath of the database to read the 
clean data and the name of the classifier file to save the trained model as a 
pickle file.

Python Script Ex: 

python models/train_cassifier.py data/DisasterResponse.db classifier.pkl


train_classifier.py outputs accuracy, recall, precision and f1-scores for each
categories. If you want to see the performance of different machine learning 
models in MultiOutputClassifier, uncomment necessary part in the main()
function. 

Default code does not make grid search. In case you want to make 
grid search, change cv_search variable to 1 in main() function. Optimal values 
of hperparameters fed to pipeline in default.

Run run.py to deploy the result. It extracts the data from SQLite database and
create data visualization. Plotly is used for data visualization.








