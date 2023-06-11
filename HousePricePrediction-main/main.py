import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def home():
    locations = sorted(data['location'].unique())
    return render_template('homepage.html', locations=locations)

@app.route('/home')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    bhk = float(bhk)
    bath = float(bath)

    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input)[0] * 1e5

    return str(np.round(prediction, 2))

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    print(f"Username: {username}, Password: {password}")

    # Add your login authentication logic here

    return "Login successful"  # Replace with appropriate response


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
