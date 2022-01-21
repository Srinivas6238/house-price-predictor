import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv('Housing.csv')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    area = request.form.get('area')
    bedroom = request.form.get('bedroom')
    bathrooms = request.form.get('bathrooms')
    stories = request.form.get('stories')
    mainroad = request.form.get('mainroad')
    guestroom = request.form.get('guestroom')
    basement = request.form.get('basement')
    hotwaterheating = request.form.get('hotwaterheating')
    airconditioning = request.form.get('airconditioning')
    parking = request.form.get('parking')
    prefarea = request.form.get('prefarea')
    furnishingstatus = request.form.get('furnishingstatus')
    output = pd.DataFrame([[area, bedroom, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating,
                            airconditioning, parking, prefarea, furnishingstatus]],
                          columns=['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement',
                                   'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus'])
    prediction = model.predict(output)[0]
    return render_template('index.html', prediction_text=f'Predicted Price is Rs. {np.round(prediction, 2)} /-')


if __name__ == '__main__':
    app.run(debug=True,port=5001)