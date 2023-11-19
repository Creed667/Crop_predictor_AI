from flask import Flask, request, render_template,url_for, redirect
import numpy as np
import pandas as pd
import sklearn
import joblib
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt


model = joblib.load('model.joblib')
scaler_minmax = joblib.load('minmaxscaler.joblib')
scaler_standard = joblib.load('standscaler.joblib')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/aboutus')
def aboutus():
    return render_template("aboutus.html")

@app.route('/base')
def base():
    return render_template("base.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/loggedin', methods = ['POST', 'GET'])
def loggedin():
    username = request.form['username']
    password = request.form['password']
    if username == "botheads" and password =="12345":
        return render_template("base.html")
    else:
        msg = "Wrong credentials. Try again!"
        return render_template("login.html", msg = msg)



@app.route("/predict", methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = scaler_minmax.transform(single_pred)
    final_features = scaler_standard.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivate".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('base.html', result=result)

# python main
if __name__ == "__main__":
    app.run(debug=True)
