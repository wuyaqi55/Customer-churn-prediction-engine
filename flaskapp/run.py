# import packages
from flask import Flask, jsonify, request, render_template
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, date


# self-defined function
def diff_month(x):
    d1 = date.today()
    d2 = datetime.strptime(x,'%Y-%m-%d')
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def object2timestamp(x):
    # return time into hour during the day    
    return datetime.strptime(x,'%H:%M:%S').timetuple().tm_hour

# model.pkl
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        age = request.form["age"]
        gender = request.form["gender"]
        security_no = request.form['security_no']
        region_category = request.form['region_category']
        membership_category = request.form['membership_category']
        joining_date = request.form['joining_date']
        joined_through_referral = request.form['joined_through_referral']
        referral_id = request.form['referral_id']
        preferred_offer_types = request.form['preferred_offer_types']
        medium_of_operation = request.form['medium_of_operation']
        internet_option = request.form['internet_option']
        last_visit_time = request.form['last_visit_time']
        days_since_last_login = request.form['days_since_last_login']
        avg_time_spent = request.form['avg_time_spent']
        avg_transaction_value = request.form['avg_transaction_value']
        avg_frequency_login_days = request.form['avg_frequency_login_days']
        points_in_wallet = request.form['points_in_wallet']
        used_special_discount = request.form['used_special_discount']
        offer_application_preference = request.form['offer_application_preference']
        past_complaint = request.form['past_complaint']
        complaint_status = request.form['complaint_status']
        feedback = request.form['feedback']
        
  
        # last_visit_time  = datetime.strptime(last_visit_time,'%H:%M:%S').timetuple().tm_hour
        
        # d1 = date.today()
        # d2 = datetime.strptime(joining_date,'%Y-%m-%d')
        # joining_date = (d1.year - d2.year) * 12 + d1.month - d2.month


        X = pd.DataFrame([[age,gender,security_no,region_category,membership_category,joining_date,
                       joined_through_referral, referral_id, preferred_offer_types, medium_of_operation, 
                       internet_option, last_visit_time, days_since_last_login,avg_time_spent, 
                       avg_transaction_value, avg_frequency_login_days,points_in_wallet, used_special_discount,
                       offer_application_preference, past_complaint, complaint_status, feedback]])

        X.columns = ['age', 'gender', 'security_no', 'region_category','membership_category', 'joining_date', 'joined_through_referral',
                     'referral_id', 'preferred_offer_types', 'medium_of_operation','internet_option', 'last_visit_time', 'days_since_last_login',
                     'avg_time_spent', 'avg_transaction_value', 'avg_frequency_login_days','points_in_wallet', 'used_special_discount',
                     'offer_application_preference', 'past_complaint', 'complaint_status','feedback']
        
        # feature engineering
        X['days_since_last_login'] = X.days_since_last_login.map(lambda x: np.nan if int(x) < 0 else int(x))
        X['avg_time_spent'] = X.avg_time_spent.map(lambda x: np.nan if float(x) < 0 else float(x))
        X['avg_frequency_login_days'] = X.avg_frequency_login_days.map(lambda x: np.nan if x == 'Error' or float(x) < 0 else float(x))
        X['points_in_wallet'] = X.points_in_wallet.map(lambda x: np.nan if float(x) < 0 else float(x))
        X['membership_period_month'] = X.joining_date.map(diff_month)
        X['last_visit_time'] = X.last_visit_time.map(object2timestamp)

        pred = model.predict(X)
    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
