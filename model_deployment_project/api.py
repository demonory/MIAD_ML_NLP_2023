#!/usr/bin/python

from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from m09_model_deployment import predict_price
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app, 
    version='1.0', 
    title='Price Prediction API Equipo 10',
    description='Price Prediction API Equipo 10')

ns = api.namespace('predict', 
     description='Pricing Regression')
   
parser = api.parser()

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
    help='Milage of the car', 
    location='args')

parser.add_argument(
    'State', 
    type=str, 
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
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_proba(args['Year','Mileage','State','Make','Model'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
