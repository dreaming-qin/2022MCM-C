
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
def VaR(data,c=0.7):
    data=data if len(data)<=31 else data[len(data)-31:]
    sub=np.array([data[i+1]-data[i] for i in range(len(data)-1)])
    sub_sort=np.sort(sub,axis=0)
    return sub_sort[int(len(sub)*(1-c))]

def format_date(df):
    for i in range(df.shape[0]):
        index=df.iloc[i,0].find('/')
        if index>2:
            if df.iloc[i,0][2]=='0':
                df.iloc[i,0]=df.iloc[i,0][3:]
            else:
                df.iloc[i,0]=df.iloc[i,0][2:]
    df.iloc[:,0]=pd.to_datetime(df.iloc[:,0])

    return df

def get_result(predict_days,LSTM_data,VaR_data,model,scaler,z,alpha=0.6):
    p1_bit=predict_future(predict_days,LSTM_data,model).tolist()
    p1_bit = scaler.inverse_transform(p1_bit).tolist()
    p1_bit=p1_bit[predict_days-1]/VaR_data[len(VaR_data)-1]-1
    p2_bit=VaR(VaR_data,c=0.7)/VaR_data[len(VaR_data)-1]*predict_days
    f_bit=alpha*p1_bit+(1-alpha)*p2_bit-z
    return f_bit

# 0是买黄金，1是卖黄金，2是买比特币，3是卖比特币
#按顺序返回is_bit_buy,is_gold_buy,status
def investment(state,buy_list,status,value,date):
    is_bit_buy,is_gold_buy=False,False
    st=state
    if st==0:
        buy_list.append('buy gold {}'.format(date))
        is_gold_buy=True
        status[1]=int(status[0]*0.99/value)
        status[0]-=status[1]*value*1.01
    elif st==1: 
        buy_list.append('sell gold {}'.format(date))
        is_gold_buy=False
        status[0]+=status[1]*value*0.99
        status[1]=0
    elif st==2:
        buy_list.append('buy bit {}'.format(date))
        is_bit_buy=True
        status[2]=status[0]/value/1.02
        status[0]=0
    elif st==3:
        buy_list.append('sell bit {}'.format(date))
        is_bit_buy=False
        status[0]+=status[2]*value*0.98
        status[2]=0
    return is_bit_buy,is_gold_buy,status