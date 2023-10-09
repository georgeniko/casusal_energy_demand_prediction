# In this file the single variable LSTM model is implmented.After loading the data,
# we design a cross-validation method which is used for time series data, with a 
# rolling window of predictions. 

import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import os
import functions as f
from keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import LSTM
from keras.layers import Activation, Dense
from tensorflow.keras.layers import  Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import TimeSeriesSplit

print_plots = 0

#Load the data in desired form
path = os.path.join(os.path.expanduser('~'), 'Documents', 'for_edit', 'GERMANY_WITH_NAN.xlsx')
df = pd.read_excel(path)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.set_index('Date Time')
df.index = pd.to_datetime(df.index)
df= df.interpolate()
df_lt = df[['Actual_Load']]


# For the correct evaluation of the predictions, the application of cross-validation is useful. 
# The size of the training set must be fixed so that we do not favor subsequent checks with a 
# larger training set. For this purpose, we move the training data window by one day.

# The training of the model and the predictions procedure is repeated for 494 days so as to use 
# all the available data. This procedure is implemented as a form of cross-validation which is 
# well-established for time series cases where cutting off pieces from the training set is 
# prohibited.

horizon = 24 #horizon of the predictions
splits=494  #so as to use all the available data of the timeseries. Training set will contain 1 year old of data
tss = TimeSeriesSplit(n_splits=splits,max_train_size=24*30*12, test_size=horizon)

fold = 0
preds = []
mse=[]
rmse=[]
mae=[]
mape=[]

WINDOW_SIZE = 25 #number of previous observations to take into account
n_epochs = 100 #100 epochs on 495 slpits is a very demaning and time cosnuming task. Change these variables for a quick test. 
number_of_fold = 1
    
sc = MinMaxScaler()
data = sc.fit_transform(df_lt) #We will use scaling for proper training 
x, y = f.df_to_X_y(data, WINDOW_SIZE) #create the array of the input layer and the target of the LSTM for later
hourly_errors = np.zeros(shape=(splits,horizon))#we will test the quality of the predictions for each hour
fold_counter=0
full_real = np.zeros(shape=(splits,horizon))

for train_idx, val_idx in tss.split(x):

    if number_of_fold<=splits:

        print("Starting fold no ", number_of_fold)
        x_train , y_train = x[train_idx] , y[train_idx]
        x_test, y_test =  x[val_idx] , y[val_idx]
        x_train= np.squeeze(x_train, 2)
        y_train=np.squeeze(y_train, -1)

        x_test=np.squeeze(x_test, 2)
        y_test=np.squeeze(y_test, -1)

        if len(x_test)!=0:

            #The architecture of the LSTM model is a powerfull one. The goal of the project was to seee the impact of targeted variable selection for training rather
            #than to find the optimal model. This architecture was stable and efficient in single, resticted and full variations and was chosen in order to properly compare
            #these three and see how variable selection affects the outcome.
            
            model = Sequential()
            model.add(LSTM(48, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
            model.add(LSTM(24, activation='relu', return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
            model.compile(optimizer='adam', loss='mse')
            history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=horizon, validation_split=0.3, verbose=0,callbacks=[es])
                
            y_pred=model.predict(x_test)
            combined = np.column_stack((y_pred))
            predictions = sc.inverse_transform(combined)
            predictions=np.squeeze(predictions.transpose()[:], -1)

            y_test=y_test[:horizon]
            combined2 = np.column_stack((y_test))
            dataY_real=sc.inverse_transform(combined2)
            dataY_real=np.squeeze(dataY_real.transpose()[:], -1)

            #Calculate scores
            mse1,rmse1,mae1,mape1 = f.evaluate_performance(predictions, dataY_real)
            for i in range(horizon):
                hourly_errors[fold_counter,i]=predictions[i]-dataY_real[i]
                full_real[fold_counter,i] = dataY_real[i] 
            mse.append(mse1)
            rmse.append(rmse1)
            mae.append(mae1)
            mape.append(mape1)
        else:
            print("Maxed out the window array!")
        number_of_fold=number_of_fold+1
        fold_counter=fold_counter+1

print("Procedure is finished!")


#Aftewards we need to create the dataframes so as to store the results properly

cols = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']
hourly_evaluations= pd.DataFrame(hourly_errors,columns=cols)

mse=[]
rmse=[]
mae=[]
mape=[]
her = np.zeros(shape=(splits,4))
for i in range(horizon):
    temp = hourly_errors[0:splits,i]
    mse1 = np.square(temp).mean()
    rmse1 = np.sqrt(mse1)
    mae1 = np.abs(temp).mean()
    mape1 = np.mean(np.abs(temp/full_real[0:splits,i]))*100
    mse.append(mse1)
    rmse.append(rmse1)
    mae.append(mae1)
    mape.append(mape1)
    
hourly_scores=pd.DataFrame(list(zip(mse, rmse,mae,mape)),columns=['mse','rmse','mae','mape'])
scores=pd.DataFrame(list(zip(mse, rmse,mae,mape)),columns=['mse','rmse','mae','mape'])
scores2 = scores["mape"].mean()
full_real_df= pd.DataFrame(full_real,columns=cols)

#Save the results to new files

# scores.to_excel('LSTMs__SCORES_GER.xlsx', sheet_name='LSTMs_SCORES_GER')
# hourly_scores.to_excel('LSTMs_HOURLY_SCORES_GER.xlsx', sheet_name='LSTMr_HOURLY_SCORES_GER')
# hourly_evaluations.to_excel('LSTMs_HOURLY_ERRORS_GER.xlsx', sheet_name='LSTMs_HOURLY_ERRORS_GER')
# full_real_df.to_excel('LSTMs_HOURLY_REAL_GER.xlsx', sheet_name='LSTMs_HOURLY_REAL_GER')

if print_plots:
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.suptitle('Single LSTM - Learning Curves \n pipeline 1')
    plt.legend()
    plt.show()
