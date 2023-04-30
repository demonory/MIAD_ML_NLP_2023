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

def predict_price(Year,Mileage,State,Make,Model):
    #Load pkl file
    model_path = os.path.abspath('car_price.pkl')
    grid_search = joblib.load(model_path)
    #difine df_
    data = {'Year': [Year], 'Mileage': [Mileage], 'State':[State],'Make':[Make],'Model':[Model]}
    df_ = pd.DataFrame(data)
 
    # Create features
    # Se define State como los unicos valores de State
    State = pd.DataFrame({
        'State_x': [9, 35, 43, 5, 21, 47, 6, 4, 18, 34, 38, 40, 28, 27, 10, 3, 42, 17, 31, 44, 12, 1, 29, 14, 36, 20, 33, 49, 22, 45, 48, 19, 37, 15, 32, 24, 11, 16, 2, 23, 25, 26, 0, 46, 41, 30, 8, 13, 39, 50, 7],
        'State_y': ['FL', 'OH', 'TX', 'CO', 'ME', 'WA', 'CT', 'CA', 'LA', 'NY', 'PA', 'SC', 'ND', 'NC', 'GA', 'AZ', 'TN', 'KY', 'NJ', 'UT', 'IA', 'AL', 'NE', 'IL', 'OK', 'MD', 'NV', 'WV', 'MI', 'VA', 'WI', 'MA', 'OR', 'IN', 'NM', 'MO', 'HI', 'KS', 'AR', 'MN', 'MS', 'MT', 'AK', 'VT', 'SD', 'NH', 'DE', 'ID', 'RI', 'WY', 'DC']
    })
    # Se define Make como los unicos valores de Make
    Make = pd.DataFrame({
        'Make_x': [17, 6, 2, 5, 24, 35, 4, 8, 36, 12, 10, 14, 26, 13, 27, 23, 37, 18, 32, 7, 15, 19, 29, 20, 22, 21, 1, 30, 25, 34, 9, 0, 31, 28, 16, 3, 33, 11],
        'Make_y': ['Jeep', 'Chevrolet', 'BMW', 'Cadillac', 'Mercedes-Benz', 'Toyota', 'Buick', 'Dodge', 'Volkswagen', 'GMC', 'Ford', 'Hyundai', 'Mitsubishi', 'Honda', 'Nissan', 'Mazda', 'Volvo', 'Kia', 'Subaru', 'Chrysler', 'INFINITI', 'Land', 'Porsche', 'Lexus', 'MINI', 'Lincoln', 'Audi', 'Ram', 'Mercury', 'Tesla', 'FIAT', 'Acura', 'Scion', 'Pontiac', 'Jaguar', 'Bentley', 'Suzuki', 'Freightliner']
    })
    # Se define Model_unique como los unicos valores de Model
    data_2 = {'Model_x': [489, 448, 499, 398, 11, 59, 87, 446, 272, 101, 88, 264, 38, 169, 419, 400, 328, 129, 104, 122, 19, 146, 186, 217, 343, 40, 310, 231, 113, 491, 66, 72, 102, 518, 148, 317, 123, 92, 247, 418, 149, 341, 299, 443, 121, 417, 458, 165, 248, 368, 280, 220, 135, 393, 107, 305, 224, 469, 82, 429],
            'Model_y': ['Wrangler', 'Tahoe4WD', 'X5AWD', 'SRXLuxury', '3', 'C-ClassC300', 'CamryL', 'TacomaPreRunner', 'LaCrosse4dr', 'ChargerSXT', 'CamryLE', 'Jetta', 'AcadiaFWD', 'EscapeSE', 'SonataLimited', 'Santa', 'Outlander', 'CruzeSedan', 'Civic', 'CorollaL', '350Z2dr', 'EdgeSEL', 'F-1502WD', 'FocusSE', 'PatriotSport', 'Accord', 'MustangGT', 'FusionHybrid', 'ColoradoCrew', 'Wrangler4WD', 'CR-VEX-L', 'CTS', 'CherokeeLimited', 'Yukon', 'Elantra', 'New', 'CorollaLE', 'Canyon4WD', 'Golf', 'Sonata4dr', 'Elantra4dr', 'PatriotLatitude', 'Mazda35dr', 'Tacoma2WD', 'Corolla4dr', 'Silverado', 'TerrainFWD', 'EscapeFWD', 'Grand', 'RAV4FWD', 'Liberty4WD', 'FocusTitanium', 'DurangoAWD', 'S60T5', 'CivicLX', 'MuranoAWD', 'ForteEX', 'TraverseAWD', 'CamaroConvertible', 'Sportage2WD']}

    Model = pd.DataFrame(data_2)
    
    merged_df = pd.merge(State, df_, left_on='State_y', right_on='State')
    merged_df_2 = pd.merge(Make, merged_df, left_on='Make_y', right_on='Make')
    merged_df_3 = pd.merge(Model, merged_df_2, left_on='Model_y', right_on='Model')
    df_ = merged_df_3[['Year', 'Mileage', 'State_x', 'Make_x', 'Model_x']].rename(columns={'State_x': 'State','Make_x': 'Make','Model_x': 'Model'})

    # Make prediction
    try:
        p1 = grid_search.predict(df_)[0]
        resultado = p1
    except:
        resultado=print("Revise los datos seleccionados")

    return resultado

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app, 
    version='1.0', 
    title='Price Prediction API Equipo 10',
    description='Price Prediction API Equipo 10')

ns = api.namespace('predict', 
     description='Car Price Prediction')
   
parser = reqparse.RequestParser()

parser.add_argument(
    'Year', 
    type=str, 
    required=True, 
    help='Year of the car', 
    location='args')

parser.add_argument(
    'Mileage', 
    type=str, 
    required=True, 
    help='Mileage of the car', 
    location='args')

state_choices = ['FL', 'OH', 'TX', 'CO', 'ME', 'WA', 'CT', 'CA', 'LA', 'NY', 'PA', 'SC', 'ND', 'NC', 'GA', 'AZ', 'TN', 'KY', 'NJ', 'UT', 'IA', 'AL', 'NE', 'IL', 'OK', 'MD', 'NV', 'WV', 'MI', 'VA', 'WI', 'MA', 'OR', 'IN', 'NM', 'MO', 'HI', 'KS', 'AR', 'MN', 'MS', 'MT', 'AK', 'VT', 'SD', 'NH', 'DE', 'ID', 'RI', 'WY', 'DC']
state_field = fields.String(
    required=True, 
    description='State of the car',
    enum=state_choices
    )

parser.add_argument(
    'State', 
    type=state_field, 
    required=True, 
    help='State of the car', 
    location='args')

parser.add_argument(
    'Make', 
    type=str, 
    required=True, 
    help='Maker of the car', 
    location='args')

parser.add_argument(
    'Model', 
    type=str, 
    required=True, 
    help='Model of the car', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class CarPricePrediction(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_price(args['Year'], args['Mileage'], args['State'], args['Make'], args['Model'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
