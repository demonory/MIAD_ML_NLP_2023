#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib
import sys
import os
from sklearn.preprocessing import LabelEncoder
from flask import Flask
from flask_restx import Api, Resource, fields, reqparse
import joblib
from flask_cors import CORS
#!/usr/bin/python


# In[4]:


import re
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import missingno as msno


# In[2]:


def clean_text(text):
  # eliminar lo que no sean letras
  text = re.sub(r'[^a-zA-Z]'," ", text)
  # minúscula
  text = text.lower()
  # filtrar palabras de una letra
  words = text.split()
  filtered_words = [word for word in words if len(word) > 1]
  text = ' '.join(filtered_words)
  # Remover stopwords
  stop_words = set(stopwords.words('english'))
  filtered_words2 = [word for word in text.split() if not word in stop_words]
  text = ' '.join(filtered_words2)
  # lematización
  lem = WordNetLemmatizer()
  text = ' '.join([lem.lemmatize(w) for w in text.split()])
  return text


# In[7]:


dataTraining = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
dataTraining['clean_plot'] = dataTraining['plot'].apply(lambda x: clean_text(x))
df_2 = dataTraining['clean_plot']


def predict_gender(Year,Title,Plot):
    #Load pkl file
    model_path = os.path.abspath('gender_clf.pkl')
    grid_search = joblib.load(model_path)
    #difine df_
    data = {'Year': [Year], 'Title': [Title], 'Plot':[Plot]}
    df_ = pd.DataFrame(data)
    df_['clean_plot'] = df_['Plot'].apply(lambda x: clean_text(x))
    df_clean = df_['clean_plot']
    df_3 = df_clean._append(df_2)
    df_4 = pd.DataFrame(df_3,columns=['clean_plot'])
 
    # Create features
    
    vect2 = TfidfVectorizer(lowercase=False)
    X_dtm2 = vect2.fit_transform(df_4['clean_plot'])
  
    p1 = grid_search.predict_proba(X_dtm2)
    resultado = p1[0]
    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
    res = dict(zip(cols,resultado))
    return res

# In[ ]:


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app, 
    version='1.0', 
    title='Movie Gender Prediction API Team 10',
    description='Movie Gender Prediction API')

ns = api.namespace('predict', 
     description='Movie Gender Prediction')
   
parser = reqparse.RequestParser()

parser.add_argument(
    'Year', 
    type=str, 
    required=True, 
    help='Year of the movie', 
    location='args')

parser.add_argument(
    'Title', 
    type=str, 
    required=True, 
    help='Title of the movie', 
    location='args')

parser.add_argument(
    'Plot', 
    type=str, 
    required=True, 
    help='Plot of the movie', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class MovieGenderPrediction(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_gender(args['Year'], args['Title'], args['Plot'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

