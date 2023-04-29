#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
from sklearn.preprocessing import LabelEncoder

def predict_price(Year,Mileage,State,Make,Model):

    grid_search = joblib.load(os.path.dirname(__file__) + '/car_price.pkl') 

    data = {'Year': [Year], 'Mileage': [Mileage], 'State':[State],'Make':[Make],'Model':[Model]}
    df_ = pd.DataFrame(data)
    le = LabelEncoder ()
  
    # Create features
    df_['State'] = le.fit_transform(df_['State'].astype('str'))
    df_['Make'] = le.fit_transform(df_['Make'].astype('str'))
    df_['Model'] = le.fit_transform(df_['Model'].astype('str'))

    # Make prediction
    p1 = grid_search.predict_price(df_)[0,1]

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add Values')
        
    else:

        Year = sys.argv[1]
        Mileage = sys.argv[2]
        State = sys.argv[3]
        Make = sys.argv[4]
        Model = sys.argv[5]
        p1 = predict_price(Year,Mileage,State,Make,Model)
        
        print(Year,Mileage,State,Make,Model)
        print('Pricing predicted: ', p1)
        