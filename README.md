# Disaster Response Machine Learning Pipeline Project

## Libraries
Primary libraries are nltk, pandas, sqlalchemy, and sklearn options for feature extraction and pipelines.
For the web app, Flask and plotly are the main libraries.

## Motivation
The goal of this project is to classify messages sent during a disaster to one of 36 categories so that help can be
sent as soon as possible if needed.

## Files

There are three main files:

**models/train_classifier.py**

Python script that classifies text messages into categories.

**data/process_data.py**

Python script that loads data, cleans it up, and then saves it to a SQL database.

**app/run.py**

Creates the Plotly charts used in the web app.

## Results
Created a multi-ouptut text classifier that runs as a web app, which would enable aid organizations to quickly process
social media messages during natural disasters to target their responses to those most in need.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
