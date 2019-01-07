# DisasterResponse

Classification of messages sent/posted on social media during disaster events.
The data is from Figure Eight to build a model for an API that classifies messages.

The model consists of 2 files:
- process_data.py
- train_classifier.py

process_data.py takes three inputs from keyboard;

- the filepath to the disaster messages
- the filepath to the categories messages
- filepath to the database to save the cleaned data

train_classifier takes 2 inputs fron the keyboard;

- the filepath of the database to read the data
- the name of the classifierfile to save the traine model as a pickle file


