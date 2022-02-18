import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import  pandas as pd
import  os
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler


dataframe = pd.read_csv('../BCHAIN-MKPRU.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values

plt.figure()
plt.plot(dataset,'g-',label='dwell')
plt.legend(loc='best')
plt.savefig('1.png')
plt.close()

# 将整型变为float
dataset = dataset.astype('float32')
#归一化 在下一步会讲解
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.9)
trainlist = dataset[:train_size]
testlist = dataset[train_size:]

def create_dataset(dataset, timestep):
#这里的look_back与timestep相同
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
model.add(LSTM(4, input_shape=(None,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=100, verbose=2)
# model.save(os.path.join("DATA","Test" + ".h5"))
# make predictions

model = load_model(os.path.join("DATA","Test" + ".h5"))
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#反归一化
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict).tolist()
# testY = scaler.inverse_transform(testY)

trainlist=scaler.inverse_transform(trainlist).tolist()

for i in range(len(testPredict)):
    trainlist.append(testPredict[i])

plt.figure()
plt.plot(trainlist,'g-',label='dwell',color='red')
plt.legend(loc='best')
plt.savefig('2.png')



