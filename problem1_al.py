
from torch import dropout
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import  pandas as pd
import  os
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout


def predict_future(days,data,model):
    ans=[0 for i in range(days)]
    var1=np.reshape(data,(len(data)))
    for i in range(days):
        ans[i]=model.predict(np.reshape(var1,(1,len(var1),-1)))
        var1=var1.tolist()
        var1.append(ans[i][0][0])
        var1=np.delete(var1,0)
    return np.reshape(ans,(len(ans),1)) 

def create_dataset(dataset, timestep):
    dataX, dataY = [], []
    for i in range(len(dataset)-timestep-1):
        a = dataset[i:(i+timestep)]
        dataX.append(a)
        dataY.append(dataset[i + timestep])
    return np.array(dataX),np.array(dataY)

def get_train_and_test_set(dataset,train_size=1000,timestep=30):
    # 获得训练集和测试集
    # train_size =1000
    trainlist = dataset[:train_size]
    # 改
    # testlist=dataset[30:]
    testlist = dataset[train_size:]
    #训练数据太少 timestep并不能过大
    trainX,trainY  = create_dataset(trainlist,timestep)
    testX,testY = create_dataset(testlist,timestep)
    return trainX,trainY,testX,testY

def get_LSTM_model():
    model = Sequential()
    # model.add(Dropout(0.2, input_shape=(None,1,)))
    model.add(LSTM(6, input_shape=(None,1)))
    model.add(Dense(1,activation='elu'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# data是1*n的ndarray向量
def VaR(data,alpha=0.7):
    data=data if len(data)<=101 else data[len(data)-101:]
    sub=np.array([data[i+1]-data[i] for i in range(len(data)-1)])
    sub_sort=np.sort(sub,axis=0)
    return sub_sort[int(len(sub)*(1-alpha))]

def format_date(df):
    for i in range(df.shape[0]):
        index=df.iloc[i,0].find('/')
        if index>2:
            if df.iloc[i,0][2]=='0':
                df.iloc[i,0]=df.iloc[i,0][3:]
            else:
                df.iloc[i,0]=df.iloc[i,0][2:]

    return df
        
    