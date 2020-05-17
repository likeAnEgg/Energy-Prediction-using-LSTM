from sklearn import preprocessing
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop,Adam
from keras import regularizers
def root_mean_squared_error(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true)))
def make_model(input_dim=10,metrics=root_mean_squared_error,loss='mse', optimizer="rmsprop",drop_rate=0.5):

  model = Sequential()
  model.add(LSTM(128,return_sequences=True, input_shape=(None,input_dim)))
  model.add(Dropout(drop_rate))
  model.add(BatchNormalization())
  model.add(LSTM(128,return_sequences=False))
  model.add(BatchNormalization())
  model.add(Dropout(drop_rate))
  model.add(Dense(1))
  model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
  
  return model
def run_model(model,x_train,y_train,epochs=50,batch_size=500,verbose=1,validation_data=(x_val,y_val),callbacks =None):
  x_train = x_train.values[:]
  x_train= x_train.reshape((x_train.shape[0],1,x_train.shape[-1]))
  y_train = np.log1p(y_train)
  if validation_data != None:
    x_val = validation_data[0].values[:]
    x_val = x_val.reshape((x_val.shape[0],1,x_val.shape[-1]))
    y_val = np.log1p(validation_data[-1])
      
  return model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=verbose,validation_data=(x_val,y_val),callbacks=callbacks)
#best_model_file = "my_model.h5"
#mc = ModelCheckpoint(best_model_file, monitor='val_loss', mode='auto',verbose=True, save_best_only=True)
es = EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0.0001, patience=5, verbose=True, mode='auto')
model = make_model(input_dim=x_train.shape[-1],drop_rate=0.2)
model.summary()

history = run_model(model,x_train,y_train,epochs=30,batch_size=500,verbose=1,validation_data=(x_val,y_val), callbacks =[es]) # callbacks =[mc, es]
loss = history.history
loss.keys()

#rmse loss
rmse_loss_train = loss['root_mean_squared_error']
rmse_loss_val = loss['val_root_mean_squared_error']
epochs_stops = es.stopped_epoch +1 # epochs number from early stopping
epochs = range(1,epochs_stops + 1)  #len(loss_train)
plt.figure(figsize=(12,6))
plt.plot(epochs,rmse_loss_train,'r', label='RMSE train loss')
plt.plot(epochs,rmse_loss_val,'b',label='RMSE val loss')
plt.title(' root mean square error loss')
plt.legend()
plt.show()

'''
submit = pd.read_csv('./input/ashrae-energy-prediction/sample_submission.csv') 
x_test = BTW_test[['meter', 'building_id', 'primary_use', 'Month', 'Day','air_temperature', 'wind_speed', 'precip_depth_1_hr', 'cloud_coverage','square_feet']]
x_test = x_test.values[:]
x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[-1]))
prediction = history.predict(x_test)
prediction = np.expm1(prediction)
submit['meter_reading'] = prediction
submit.to_csv('submission.csv', index=False,float_format='%.4f')
'''