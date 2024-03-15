# *********** Importing Libraries ***********
import pandas as pd 
import numpy as np
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt1
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
import warnings
import os

# *********** Ignore Warnings ***********

warnings.filterwarnings('ignore')
print('hello world')

# *********** Flask App ***********

# Flask web server instance is created and the static_folder is set to build, so that the react app build files can be served from the flask server.
app = Flask(__name__, static_folder='build') 
CORS(app) # Cross Origin Resource Sharing

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path): # This function checks if the requested path exists in static folder and serves the react app build files 
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        print('Line 19')
        return send_from_directory(app.static_folder, path)
    else:
        print('Line 22')
        return send_from_directory(app.static_folder, 'index.html')

# *********** Importing Data ***********
def get_data(ticker):
    end = datetime.now()
    start = datetime(end.year-2, end.month, end.day)
    data = yf.download(ticker, start=start, end=end)
    df = pd.DataFrame(data=data)
    df.to_csv(''+ticker+'.csv')      
    return df

# *********** Linear Regression Algorithm ***********
def linear_reg_algo(df):

# the number of days the stock price to be forecasted into the future
    
    forecast= int(10)

# Creation of another column that contains the stock price of the next 10 days    

    df['Close after n days'] = df['Close'].shift(-forecast)

# Creation of a new dataframe with only the 'Close' and 'Close after 10 days' columns    

    df_1 = df[['Close', 'Close after n days']]

# Data Prep for training and testing
    #contains the 'Close after 10 days' column values except the last 10 days
    y = np.array(df_1.iloc[:-forecast,-1]) 
    # reshaping to a 2-D array
    y = np.reshape(y, (-1,1)) 
    #contains all columns except the 'Close after 10 days' column
    X = np.array(df_1.iloc[:-forecast,0:-1]) 

# #contains last 10 days rows of all columns except the 'Close after 10 days' column

    X_forecasted = np.array(df_1.iloc[-forecast:,0:-1]) 

# Splitting the data into training and testing data (80% and 20% respectively)
    X_train = X[0:int(0.8*len(df)),:]
    X_test = X[int(0.8*len(df)):,:]
    y_train = y[0:int(0.8*len(df)),:]
    y_test = y[int(0.8*len(df)):,:]

# Feature Scaling : Normalizing the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    X_forecasted = sc.transform(X_forecasted)

# Training the model
    regressor = LinearRegression(n_jobs=-1)
    regressor.fit(X_train, y_train)
# Testing the model
    y_test_pred = regressor.predict(X_test)
    y_test_pred = y_test_pred*(1.02)

# Plotting    
    fig = plt1.figure(figsize=(7,7))
    plt1.plot(y_test, color = 'red', label = 'Real Stock Price')
    plt1.plot(y_test_pred, color = 'blue', label = 'Predicted Stock Price')
    plt1.title('Stock Price Prediction')
    plt1.xlabel('Time')
    plt1.ylabel('Stock Price')
    plt1.legend(loc='upper left')
    plt1.savefig('linear_reg_algo.png')



# Forecasting the future stock price
    forecast_set = regressor.predict(X_forecasted)
    forecast_set = forecast_set*(1.02)
    mean = forecast_set.mean()
    lr_pred = forecast_set[0,0]
    return df, lr_pred, forecast_set, mean


def signal(today_stock, mean, default="NO_ORDER"):
    order_type = default
    if today_stock['Close'] < mean:
        order_type = "BUY"
        print()
    else:
        order_type = "SELL"
    
    return {'order_type': order_type}

# Tells flask  to trigger this function when the /predict endpoint is hit with a GET request.
@app.route('/predict', methods=['GET'])

# retrieves the ticker parameter from the query string of GET request.  Initiates the signal function to predict the order type.

def predict_stock():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'error': 'Missing ticker parameter'}), 400
    
    # Initiates the get_data function to download the stock data and save it as a csv file.
    get_data(ticker)
    # Reads the csv file and stores it in a pandas dataframe. 
    df = pd.read_csv(''+ticker+'.csv')
    today_stock = df.iloc[-1]
    # Initiates the linear_reg_algo function to predict the stock price.

    df, lr_pred, forecast_set, mean = linear_reg_algo(df)
    decision = signal(today_stock, mean)

    # Returns the predicted stock price and the order type as a JSON response.
    result = {'ticker': ticker,
              'today_stock': today_stock.to_dict(),
              'lr_pred' : lr_pred,
              'decision': decision,
              'forecast_set': forecast_set.tolist(),
              }
    return jsonify(result)

# Tells flask to trigger this function when the /get_image endpoint is hit with a GET request.

@app.route('/get_image')
def get_image():
    return send_file('linear_reg_algo.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

