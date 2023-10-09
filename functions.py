import numpy as np


#functions df_to_X_y and df_to_X_y2 were seen on: "LSTM Time Series Forecasting Tutorial in Python" [YT], Greg Hogg, 
# url: https://colab.research.google.com/drive/1HxPsJvEAH8L7XTmLnfdJ3UQx7j0o1yX5?usp=sharing

#Their goal is to create the input array of the input layer of the LSTM as well as to create the labels
#target variable for proper training. The window size determines the length of the pervious observations
#to take into account.

def df_to_X_y(df, window_size=6):
  df_as_np = df#.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)


def df_to_X_y2(df, window_size=6):
    df_as_np = df#.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size][0]#[1]
        y.append(label)
    return np.array(X), np.array(y)



#This function returns some basic metrics for the performance evaluation of the model

def evaluate_performance(predictions, actual):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    mape = np.mean(np.abs(errors/actual))*100
    return mse,rmse,mae,mape

