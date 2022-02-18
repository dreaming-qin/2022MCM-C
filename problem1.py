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

# 改
# df = pd.read_csv('../LBMA-GOLD.csv', engine='python', skipfooter=3)
df = pd.read_csv('../BCHAIN-MKPRU.csv', engine='python', skipfooter=3)

dataset = df.iloc[:,1].values
dataset=np.reshape(dataset,(len(dataset),-1))

# 将整型变为float
dataset = dataset.astype('float32')
#归一化 在下一步会讲解
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset)*0.9)
# train_size =1000
trainlist = dataset[:train_size]
# 改
testlist=dataset[100:]
# testlist = dataset[train_size:]

def create_dataset(dataset, timestep):
    dataX, dataY = [], []
    for i in range(len(dataset)-timestep-1):
        a = dataset[i:(i+timestep)]
        dataX.append(a)
        dataY.append(dataset[i + timestep])
    return np.array(dataX),np.array(dataY)
#训练数据太少 timestep并不能过大
timestep = 100
trainX,trainY  = create_dataset(trainlist,timestep)
testX,testY = create_dataset(testlist,timestep)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1] ,1 ))

# create and fit the LSTM network
model = Sequential()
# model.add(Dropout(0.2, input_shape=(None,1,)))
model.add(LSTM(12, input_shape=(None,1),dropout=0.05))
model.add(Dense(1,activation='elu'))
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型
# 改
# model = load_model(os.path.join("DATA","LSTM_gold" + ".h5"))
# model = load_model(os.path.join("DATA","LSTM_bit" + ".h5"))
model.fit(trainX, trainY, epochs=500, batch_size=100, verbose=2)
# a=1
# 改
model.save(os.path.join("DATA","LSTM_bit" + ".h5"))
# model.save(os.path.join("DATA","LSTM_gold" + ".h5"))

# make predictions

testPredict = model.predict(testX)

#反归一化，获得结果图片

testPredict = scaler.inverse_transform(testPredict).tolist()
value=dataset[:len(dataset)-len(testPredict)]
value=scaler.inverse_transform(value).tolist()
value=value+testPredict

date=df.iloc[:len(value),0].values.tolist()

real_value=scaler.inverse_transform(dataset).tolist()

dic={'Date':np.reshape(date,(len(date))),'predict value':np.reshape(value,(len(value))),
'real value':np.reshape(real_value,(len(real_value))) }
# 改
# pd.DataFrame(dic).to_csv('../result/黄金趋势(100天后).csv',index=False)
pd.DataFrame(dic).to_csv('../result/比特币趋势(100天后).csv',index=False)



# plt.figure()
# plt.plot(trainlist,'g-',label='dwell',color='red')
# plt.legend(loc='best')
# plt.savefig('2.png')




