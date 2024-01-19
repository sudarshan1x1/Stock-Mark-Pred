# *********** Importing Libraries ***********
import pandas as pd 
import numpy as np
import datetime as dt 
import yfinance as yf
import matplotlib.pyplot as plt1
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 

# *********** Ignore Warnings ***********
import warnings
warnings.filterwarnings('ignore')

# *********** Importing Data ***********
def get_data(ticker):
    end = dt.datetime.now()
    start = dt.datetime(end.year-2, end.month, end.day)
    data = yf.download(ticker, start=start, end=end)
    df = pd.DataFrame(data=data)
    df.to_csv(''+ticker+'.csv')
    return

# *********** Linear Regression Algorithm ***********
def linear_reg_algo(df):

# the number of days the stock price to be forecasted into the future
    forecast= int(10)
# Creation of another column that contains the stock price of the next 10 days    
    df['Close after n days'] = df['Close'].shift(-forecast)
# Creation of a new dataframe with only the 'Close' and 'Close after 10 days' columns    
    df_1 = df[['Close', 'Close after n days']]
# Data Prep for training and testing
    y = np.array(df_1.iloc[:-forecast,-1])
    y = np.reshape(y, (-1,1))

    X = np.array(df_1.iloc[:-forecast,0:-1])
# Forecasting the next 10 days
    X_forecasted = np.array(df_1.iloc[-forecast:,0:-1])
# Splitting the data into training and testing data
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
    plt1.show()
    plt1.close(fig)



# Forecasting the future stock price
    forecast_set = regressor.predict(X_forecasted)
    forecast_set = forecast_set*(1.02)
    mean = forecast_set.mean()
    lr_pred = forecast_set[0,0]
    print()
    print( "Tomorrow's "+ticker+" Closing Price Prediction by Linear Regression: ",lr_pred)
    return df, lr_pred, forecast_set, mean


def signal(today_stock, mean):
    if today_stock['Close'] < mean:
        order_type = "BUY"
        print()
    else:
        order_type = "SELL"
    
    return order_type



ticker = input("Enter the stock ticker: ").upper()

get_data(ticker)
df = pd.read_csv(''+ticker+'.csv')
today_stock = df.iloc[-1]
print("Today's "+ticker+" Stock Data : ", today_stock)
print("**********************************************")

df, lr_pred, forecast_set, mean = linear_reg_algo(df)
print("**********************************************")
decision = signal(today_stock, mean)
print("According to the Model the Signal is to ",decision," the stock.")

print(" Forecasted Stock Price for Next 10 Days:")
print(forecast_set)