#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout


# In[4]:


data=pd.read_csv('F:\Stock Price Prediction 1.csv',index_col='Date',parse_dates=True)


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


data.columns


# In[8]:


data.info()


# In[9]:


for columns in data.columns:
    plt.figure(figsize=(12,4))
    plt.title(f"Stock {columns} Price")
    plt.plot(data.index,data[columns])
    plt.xticks(rotation=45)


# In[10]:


plt.figure(figsize=(12,4))
plt.title("Stock Price")
for columns in data.columns:
    if(columns !='Volume'):
        plt.plot(data.index,data[columns],label=columns)
plt.xticks(rotation=45)
plt.legend()


# In[11]:


scaler=MinMaxScaler(feature_range=(0,1))


# In[13]:


data=data['Close']
data.shape


# In[14]:


df=scaler.fit_transform(np.array(data).reshape([data.shape[0],1]))


# In[15]:


def create_seq(data,time_step=60):
    X=[]
    y=[]
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step)])
        y.append(data[i+time_step])
    return X,y


# In[16]:


time_step=100
X,y=create_seq(df,time_step)


# In[17]:


X=np.array(X)
X=X.reshape(X.shape[0],X.shape[1],1)
y=np.array(y)
X.shape,y.shape


# In[18]:


X_train,X_test,y_train,y_test=X[:int(data.shape[0]*0.8)],X[int(data.shape[0]*0.8):],y[:int(data.shape[0]*0.8)],y[int(data.shape[0]*0.8):]
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[19]:


model=Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=X_train[0].shape))
model.add(LSTM(64,return_sequences=True))
model.add(LSTM(32))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))
model.compile(optimizer= keras.optimizers.Adam(learning_rate=0.001),loss="mean_squared_error",metrics=[keras.metrics.RootMeanSquaredError()])
model.summary()


# In[20]:


model.fit(X_train,y_train,epochs=100)


# In[21]:


trainPred=model.predict(X_train)
testPred=model.predict(X_test)
trainPred=scaler.inverse_transform(trainPred)
testPred=scaler.inverse_transform(testPred)


# In[22]:


look_back=time_step
trainPredPlot=np.empty_like(scaler.inverse_transform(df))
trainPredPlot[:]=np.nan
trainPredPlot[look_back:len(trainPred)+look_back]=trainPred
testPredPlot=np.empty_like(scaler.inverse_transform(df))
testPredPlot[:]=np.nan
testPredPlot[len(trainPred)+look_back:len(trainPred)+look_back+len(testPred)]=testPred
plt.plot(scaler.inverse_transform(df),label="Actual close price")
plt.plot(trainPredPlot,label="Training prediction close price")
plt.plot(testPredPlot,label="Predicted close price")
plt.legend()
plt.show()


# In[23]:


predection_data=np.array(data[-time_step:])
predection_data=predection_data.reshape([predection_data.shape[0],1])
def predication(data,days=30):
    data=scaler.transform(data)
    pred=[]
    for i in range(1,days+1):
        nxt_day=model.predict([data],verbose=0)
        pred.append(scaler.inverse_transform(nxt_day)[0])
        data[:-1]=data[1:]
        data[-1]=nxt_day[0]
    return np.array(pred).squeeze()
days=30
res=predication(predection_data,days)


# In[24]:


trainPredPlot=np.zeros(shape=[len(predection_data)+1+days])
trainPredPlot[:]=np.nan
trainPredPlot[len(predection_data)]=res[-1]
trainPredPlot[len(predection_data)+1:]=res
df_=predection_data
plt.plot(df_,label="Actual close price")
plt.plot(trainPredPlot,label="Predicted close price")
plt.legend()
plt.show()

