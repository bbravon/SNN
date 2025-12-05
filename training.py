import pandas as pd
import tensorflow
import process
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout,GRU
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



def pre_train(csv,seq_mode=1,stride=10):
    df=pd.read_csv(csv)
    data=df['0'].values
    labels=df['label'].values
    if seq_mode==1:
        X,y= process.sequence_generator(data,labels,window_size=240,stride=stride)
    else:
        X,y= process.sequence_generator2(data,labels,window_size=240,stride=stride)
    X=X.astype(int)
    lon=len(X)
    X_train = X[0:int(lon*0.7)]
    X_val = X[int(lon*0.7)+1:int(lon*0.8)]
    X_test = X[int(lon*0.8)+1:lon]
    y_train = y[0:int(lon*0.7)]
    y_val = y[int(lon*0.7)+1:int(lon*0.8)]
    y_test = y[int(lon*0.8)+1:lon]

    return (X_train,y_train),(X_val,y_val),(X_test,y_test)



early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

# Add these callbacks to model.fit()
callbacks = [early_stopping, lr_scheduler]

def train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32,callbacks=[early_stopping,lr_scheduler]):

    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=4, activation='sigmoid', padding='same'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(128,return_sequences=True,recurrent_activation='sigmoid')),
        Bidirectional(LSTM(64,return_sequences=True,recurrent_activation='sigmoid')),
        Bidirectional(LSTM(32, return_sequences=False,recurrent_activation='sigmoid')),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    
    optimizer = RMSprop(learning_rate=0.001)
    
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae'])
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),callbacks=callbacks)

    return model

  

    