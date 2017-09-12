# -*- coding: utf-8 -*-
"""
Script to run stock predictions based on a recurrent neural network (rNN)
and a Kalman filter (KF).

In addition, a "stock game" is included as well, where the performance of 
an investment strategy is analyzed

(C) Ruben Cubo, 2017-09-07
"""

# Import libraries to be used

import quandl
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

def diff_returns(dataset, interval=1):
    # This function gives the simple return over some interval 
    # Input:
    #    dataset: some stock prices (open, close, adjusted...)
    #    interval: interval from which the returns will be computed
    # Output:
    #    A pandas series with the returns
   
	 diff = list()
	 for i in range(interval, len(dataset)):
 	   value = (dataset[i] - dataset[i - interval])/dataset[i - interval]
	 diff.append(value)
	 return pd.Series(diff)

def values_from_returns(x0, dataset, interval=1):
    # This function gives the stock value over a simple return list 
    # Input:
    #    x0: The initial value of the stock
    #    dataset: some returns from a stock (daily, yearly...)
    #    interval: interval from which the stock values will be computed
    # Output:
    #    A numpy array with the stock values
   
    val = np.array([x0])
    for k in range(0,len(dataset)):
      val = np.append(val,(1+dataset[k])*val[k])
    return val

def split_train_test(features, labels, index):
    # This function splits a list of features and labels into a training and a test set 
    # Input:
    #    features: some features
    #    labels: some labels
    #    index: the splitting point between the training set and the test set
    # Output:
    #    DataFrames of the correct dimension
    
    features = features.reshape(features.shape[0],1,features.shape[1])
    features_train = features[0:index]
    labels_train = labels[0:index]
    features_test = features[index:]
    labels_test = labels[index:]
    return features_train, labels_train, features_test, labels_test

def pred_step(model, split, features, labels, labels_scaler, to_predict=1):
    # Predicts the next stock value using data from a rNN 
    # Input:
    #    model: the rNN model
    #    split: the point in time where the dataset is split between training and test set
    #    features, labels: the stock features and the prices
    #    label_scaler: the object used to scale the labers
    #    to_predict: the prediction horizon (default 1)
    # Output:
    #    Predicted stock value non-normalized
    
    features_train_r,labels_train,features_test_r,labels_test = split_train_test(features, labels, split)
        
    model.load_weights('google_first_test.hdf5') #load the weights
    
    pred_values = model.predict(features_test_r[:to_predict])
    pred_values_unnormalized = labels_scaler.inverse_transform(pred_values)
    
    return pred_values_unnormalized

def kalman_filter(A,C,Q,R,P_tm1tm1,x_tm1tm1,yt):
    # Implements the Kalman Filter for a linear system without exogenous input
    # Input:
    #    A, C: The system dynamics and the relation states-output
    #    Q, R: The covariance matrices for the process and measurement noises
    #    P_tm1tm1: The covariance matrix of the system at time t-1
    #    x_tm1tm1: The state vector of the system at time t-1
    #    yt: Measurement at time t
    # Output:
    #    x_ttm1: Prediction using data only up to t-1
    #    x_tt: Filtered states using both x_ttm1 and the new measurement yt
    #    P_t: The covariance matrix of the system at time t
    
    # Prediction step
    x_ttm1 = np.dot(A,x_tm1tm1)
    
    # Update step
    P_ttm1 = np.dot(np.dot(A,P_tm1tm1),A.T) + Q
    K_t = np.dot(P_ttm1,np.dot(C.T,(np.dot(C,np.dot(P_ttm1,C.T)) + R)**-1))
    x_tt = np.dot(A,x_tm1tm1) + np.dot(K_t,(yt - np.dot(C,np.dot(A,x_tm1tm1))))
    P_t = P_ttm1 - np.dot(K_t,np.dot(C,P_ttm1))
    
    return x_ttm1, x_tt, P_t
   

quandl.ApiConfig.api_key = "chFkPXYX2M3NpfitPZhu" # Your Quandl API key goes here
    
np.random.seed(2001) # Fix the random seed to replicate results

# Load the data (can be exchanged for anything with the same structure, i.e. other stocks)
google_data = quandl.get("WIKI/GOOGL") 

# Extract the features
google_data_features = google_data.iloc[1:] 

# Plot the correlations between the features
sns.heatmap(google_data_features.corr())
plt.savefig('Correlations.pdf')
plt.show()

# Plot the adjusted close stock prices
plt.plot(google_data.index,google_data['Adj. Close'])
plt.savefig('AdjClosing.pdf')
plt.show()

# Split Ratio does not give anything, so we take it out
google_data_features = google_data_features.drop(['Split Ratio'],axis=1)

# Transform the labels into daily returns
google_data_labels = diff_returns(google_data['Adj. Close'])

# Scale the features and the labels in the range (-1,1)
google_s_features = MinMaxScaler(feature_range=(-1,1))
google_scaled_features = google_s_features.fit_transform(google_data_features)
google_s_labels = MinMaxScaler(feature_range=(-1,1))
google_scaled_labels = google_s_labels.fit_transform(google_data_labels.values.reshape(-1,1))

# Split the dataset with 80% training and 20% testing
test_train_split = int(np.floor(0.8*google_scaled_features.shape[0]))

google_features_train_r,google_labels_train,google_features_test_r,google_labels_test = split_train_test(google_scaled_features, google_scaled_labels, test_train_split)

# rNN Model

# Import keras' relevant libraries
from keras.layers import Input, Dense, Flatten, Dropout, Activation, advanced_activations, LSTM, TimeDistributed
from keras.models import Model, model_from_json, Sequential
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

batch_size=32

# Define the model
model = Sequential()
model.add(LSTM(128, input_shape=(1,google_features_train_r.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['MSE'])
model.summary()

# We use a checkpoint to save the weights that give best validation loss
checkpointer = ModelCheckpoint(filepath='google_first_test.hdf5', 
                               verbose=2, save_best_only=True)

# Fit the model
history = model.fit(google_features_train_r, google_labels_train, epochs=500, batch_size=batch_size, verbose=2, validation_split=0.3, callbacks=[checkpointer], shuffle=False)

model.load_weights('google_first_test.hdf5') # Load the weights

# Predict values in the training set for visualization only and plot them
pred_values = model.predict(google_features_train_r)

plt.plot(google_data.index[1:test_train_split+1],google_labels_train,label='Real')
plt.plot(google_data.index[1:test_train_split+1],pred_values,label='Predicted')
plt.legend()
plt.savefig('pred_returns.pdf')
plt.show()

# Learning curve to see if we are overfitting

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.savefig('learning_curve.pdf')
plt.show()

# Use the rNN to do 1-step predictions. We use all the test set and update our
# sets to be predicted at each time step

pred_value_absolute = np.array([])
n_predictions = len(google_labels_test)-1
for k in range(0,n_predictions):
    test_train_split += 1
    pred_values_unn = pred_step(model, checkpointer, test_train_split, google_scaled_features, google_scaled_labels, google_s_labels)
    previous_value = np.array([google_data['Adj. Close'].iloc[test_train_split-1]])
    pred_value_absolute = np.append(pred_value_absolute,values_from_returns(previous_value, pred_values_unn)[1])

plt.plot(google_data.index[test_train_split-n_predictions:test_train_split],google_data['Adj. Close'].iloc[test_train_split-n_predictions:test_train_split],label='Real')
plt.plot(google_data.index[test_train_split-n_predictions:test_train_split],pred_value_absolute,label='rNN')
plt.legend()
plt.savefig('pred_rnn.pdf')
plt.show()

# Compute the MSE of the predicted values

mse_pred = ((pred_value_absolute - np.array([google_data['Adj. Close'].iloc[test_train_split-n_predictions:test_train_split]]))**2).mean()

# Kalman Filter

# Values to explore for Q and R

Qv_values = np.logspace(-5,0,num=25)
Rv = 0.1

# Define the model matrices A and C

A = np.array([[3, -3, 1],[1, 0, 0],[0, 1, 0]],dtype=np.float64)
C = np.array([[1, 0, 0]],dtype=np.float64)

# Initiate the vector to store the innovations variance

innov_var=np.array([])

# Loop and store the innovations variance

for k in Qv_values:
    
    Q = k*np.identity(3)
    R = np.array([[Rv]])    
    
    x_tm1tm1 = np.fliplr(np.array([google_data['Adj. Close'].iloc[0:3]]))
    x_tm1tm1 = x_tm1tm1.T
    Pt = np.identity(3)
    
    x_pred = np.array([])
    x_fil = np.array(x_tm1tm1)
    innov = np.array([])
    
    for k in range(0,test_train_split-n_predictions-3):
        yt = np.array([[google_data['Adj. Close'].iloc[3+k]]])
        x_predt,x_fil,Pt = kalman_filter(A,C,Q,R,Pt,x_fil,yt)
        innov = np.append(innov,yt-x_predt)
        x_pred = np.append(x_pred,x_predt[0])
    
    innov_var = np.append(innov_var,np.std(innov)**2)

# Obtain the Q that has the lowest innovation variance
    
Qv_min = Qv_values[np.argmin(innov_var)]

# Re-run with that value

Q = Qv_min*np.identity(3)
R = np.array([[Rv]])    
    
x_tm1tm1 = np.fliplr(np.array([google_data['Adj. Close'].iloc[test_train_split-n_predictions-3:test_train_split-n_predictions]]))
x_tm1tm1 = x_tm1tm1.T
Pt = np.identity(3)
    
x_pred = np.array([])
x_fil = np.array(x_tm1tm1)
innov = np.array([])
    
for k in range(0,n_predictions):
    yt = np.array([[google_data['Adj. Close'].iloc[test_train_split-n_predictions+k]]])
    x_predt,x_fil,Pt = kalman_filter(A,C,Q,R,Pt,x_fil,yt)
    innov = np.append(innov,yt-x_predt)
    x_pred = np.append(x_pred,x_predt[0])

# Plot the Kalman Filter predictions
plt.plot(google_data.index[test_train_split-n_predictions:test_train_split],google_data['Adj. Close'].iloc[test_train_split-n_predictions:test_train_split],label='Real')
plt.plot(google_data.index[test_train_split-n_predictions:test_train_split],x_pred,label='Kalman')
plt.legend()
plt.savefig('pred_kalman.pdf')
plt.show() 

# Plot all three predictions
    
plt.plot(google_data.index[test_train_split-n_predictions:test_train_split],google_data['Adj. Close'].iloc[test_train_split-n_predictions:test_train_split],label='Real')
plt.plot(google_data.index[test_train_split-n_predictions:test_train_split],pred_value_absolute,label='rNN')
plt.plot(google_data.index[test_train_split-n_predictions:test_train_split],x_pred,label='Kalman')
plt.legend()
plt.savefig('pred_rnn_kalman.pdf')
plt.show()   

# Compute the MSE

mse_kalman =  ((x_pred - np.array([google_data['Adj. Close'].iloc[test_train_split-n_predictions:test_train_split]]))**2).mean()

# Print the MSEs and the innovation variance in the console
print('Innovation variance for T=', -np.log(Qv_min/Rv), 'is:\n', np.std(innov)**2)

print('MSE for', n_predictions, "predictions is:\n")
print('rNN:', mse_pred)
print('Kalman Filter:', mse_kalman)


# Now, the money game!

money_initial = 1000.0
transaction_fee = 0 # We assume no transaction fees (can be changed)
initial_ratio = 0.8 # Fraction of money we start investing

# Buy and hold strategy

money_bh = np.array([money_initial])
bh_returns = diff_returns(google_data['Adj. Close'].iloc[test_train_split-n_predictions-1:test_train_split])
for k in range(0,len(bh_returns)):
    money_bh = np.append(money_bh,money_bh[k]*(1+bh_returns[k]))
    
# Estimations using KF
    
money_kalman = np.array([initial_ratio*money_initial]) # We start with some of our money
money_kalman_remaining = (1-initial_ratio)*money_initial
money_kalman_total = np.array([money_initial])
perc_kalman_buy = 0.1 # Fraction of buying. If a price is expected to go up, we buy
perc_kalman_sell = 0.1 # Fraction of selling. If a price is expected to go up, we sell
ratio_invested_kalman = np.array([initial_ratio]) # Just to monitor how much is in the market

kalman_return = diff_returns(x_pred) # Compute the returns from the predicted values
for k in range(0,len(kalman_return)):
    if kalman_return[k] > 0: # Stock is predicted to go up, so we buy
        money_kalman_invested = perc_kalman_buy*money_kalman_remaining
        money_kalman_remaining -= money_kalman_invested+transaction_fee
        money_kalman_close = money_kalman_invested+money_kalman[k]
        money_kalman = np.append(money_kalman,money_kalman_close*(1+bh_returns[k]))
        money_kalman_total = np.append(money_kalman_total,money_kalman[-1] + money_kalman_remaining)
        ratio_invested_kalman = np.append(ratio_invested_kalman,money_kalman[-1]/money_kalman_total[-1])
    elif kalman_return[k] < 0: # Stock is predicted to go down, so we sell
        money_kalman_sold = perc_kalman_sell*money_kalman[k]
        money_kalman_remaining += money_kalman_sold-transaction_fee
        money_kalman_close = money_kalman[k]-money_kalman_sold
        money_kalman = np.append(money_kalman,money_kalman_close*(1+bh_returns[k]))
        money_kalman_total = np.append(money_kalman_total,money_kalman[-1] + money_kalman_remaining)
        ratio_invested_kalman = np.append(ratio_invested_kalman,money_kalman[-1]/money_kalman_total[-1])
    else: # Stock is predicted to stay as it is, so we don't do anything
        money_kalman = np.append(money_kalman,money_kalman[k])
        money_kalman_total = np.append(money_kalman_total,money_kalman[-1] + money_kalman_remaining)
        ratio_invested_kalman = np.append(ratio_invested_kalman,ratio_invested_kalman[-1])

# Estimations using rNN
    
money_rnn = np.array([initial_ratio*money_initial]) # We start with some of our money
money_rnn_remaining = (1-initial_ratio)*money_initial
money_rnn_total = np.array([money_initial])
perc_rnn_buy = 0.1 # Fraction of buying. If a price is expected to go up, we buy
perc_rnn_sell = 0.1 # Fraction of selling. If a price is expected to go up, we sell
ratio_invested_rnn = np.array([initial_ratio])
rnn_return = diff_returns(pred_value_absolute)
for k in range(0,len(rnn_return)):
    if rnn_return[k] > 0: # Stock is predicted to go up, so we buy
        money_rnn_invested = perc_rnn_buy*money_rnn_remaining
        money_rnn_remaining -= money_rnn_invested+transaction_fee
        money_rnn_close = money_rnn_invested+money_rnn[k]
        money_rnn = np.append(money_rnn,money_rnn_close*(1+bh_returns[k]))
        money_rnn_total = np.append(money_rnn_total,money_rnn[-1] + money_rnn_remaining)
        ratio_invested_rnn = np.append(ratio_invested_rnn,money_rnn[-1]/money_rnn_total[-1])
    elif rnn_return[k] < 0: # Stock is predicted to go down, so we sell
        money_rnn_sold = perc_rnn_sell*money_rnn[k]
        money_rnn_remaining += money_rnn_sold-transaction_fee
        money_rnn_close = money_rnn[k] - money_rnn_sold
        money_rnn = np.append(money_rnn,money_rnn_close*(1+bh_returns[k]))
        money_rnn_total = np.append(money_rnn_total,money_rnn[-1] + money_rnn_remaining)
        ratio_invested_rnn = np.append(ratio_invested_rnn,money_rnn[-1]/money_rnn_total[-1])
    else: # Stock is predicted to stay as it is, so we don't do anything
        money_rnn = np.append(money_rnn,money_rnn[k])
        money_rnn_total = np.append(money_rnn_total,money_rnn[-1] + money_rnn_remaining)
        ratio_invested_rnn = np.append(ratio_invested_rnn,ratio_invested_rnn[-1])

# Plot the results
        
plt.plot(google_data.index[test_train_split-n_predictions-1:test_train_split],money_bh,label='Buy & Hold')
plt.plot(google_data.index[test_train_split-n_predictions:test_train_split],money_kalman_total,label='Kalman')
plt.plot(google_data.index[test_train_split-n_predictions:test_train_split],money_rnn_total,label='rNN')
plt.legend()
plt.savefig('money_game.pdf')
plt.show()

# Plot the ratios of money invested in the market for visualization purposes

plt.plot(google_data.index[test_train_split-n_predictions:test_train_split],ratio_invested_kalman,label='Kalman')
plt.plot(google_data.index[test_train_split-n_predictions:test_train_split],ratio_invested_rnn,label='rNN')
plt.legend()
plt.savefig('money_ratios.pdf')
plt.show()