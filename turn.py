from training import *
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Reshape,
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense, GaussianNoise
)
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import backend as K
training,evaluation,testing = pre_train('data_basic_all.csv',stride=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6)

X_train, y_train = training
X_val,y_val=evaluation
X_test,y_test = testing






def monotonicity_loss3(lambda_penalty=1.0, delta_penalty=1.0, importance_penalty=1.0):
    def loss(y_true, y_pred):
        # Mean Squared Error
        mse = K.mean(K.square(y_true - y_pred))
        
        # Difference between consecutive predictions and true values
        pred_diffs = y_pred[:, 1:] - y_pred[:, :-1]
        true_diffs = y_true[:, 1:] - y_true[:, :-1]
        
        # Penalize negative differences (i.e., non-monotonic behavior)
        monotonic_penalty = K.sum(K.maximum(-K.square(pred_diffs), 0.0))
        
        # Penalize incorrect increase rates
        delta_penalty_value = K.sum(K.abs(pred_diffs - true_diffs))
        
        # Importance weighting: Higher penalty near value 1
        importance_weights = K.square(y_true)
        importance_penalty_value = K.sum(importance_weights * K.square(y_true - y_pred))
        
        # Combine MSE and penalties
        return mse + lambda_penalty * monotonic_penalty + delta_penalty * delta_penalty_value + importance_penalty * importance_penalty_value
    
    return loss




def inicio():
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=256, strides=1, kernel_size=4,activation='relu', padding='valid'),
        MaxPooling1D(),
        LSTM(128, return_sequences=True ),
        LSTM(64, return_sequences=True ),
        LSTM(32, return_sequences=False ),
        Dropout(0.3),
        Dense(16,activation='tanh'),
        Dropout(0.2),
        Dense(1),
    ])
    optimizer = RMSprop(learning_rate=0.001) 
    
    model.compile(optimizer=optimizer,
                  loss=monotonicity_loss3(lambda_penalty=1.2,delta_penalty=1,importance_penalty=0), 
                  metrics=['mse'])
    
    return model
with tf.device('/GPU:0'):
    model=inicio()
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),callbacks=[early_stopping,lr_scheduler])
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    df=pd.DataFrame(loss)
    df["val_loss"]=val_loss
    df.to_csv('loss2.csv',index=False)
    model.save('few.keras')

    
