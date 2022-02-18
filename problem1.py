from problem1_al import VaR, create_dataset, get_LSTM_model, get_train_and_test_set,predict_future
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
df = pd.read_csv('../LBMA-GOLD.csv', engine='python', skipfooter=3)
# df = pd.read_csv('../BCHAIN-MKPRU.csv', engine='python', skipfooter=3)

dataset = df.iloc[:,1].values
dataset=np.reshape(dataset,(len(dataset),-1))

# 将整型变为float
dataset = dataset.astype('float32')
#归一化 在下一步会讲解
scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

# 获得训练集和测试集
train_size = int(len(dataset)*0.9)
#训练数据太少 timestep并不能过大
timestep = 30
trainX,trainY,testX,testY = get_train_and_test_set(dataset,train_size,timestep)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1] ,1 ))

# 创建LSTM
model = get_LSTM_model()

# 训练模型
# 改
# model = load_model(os.path.join("DATA","LSTM_gold" + ".h5"))
# model = load_model(os.path.join("DATA","LSTM_bit" + ".h5"))
model.fit(trainX, trainY, epochs=4000, batch_size=100, verbose=2)
# 改
# model.save(os.path.join("DATA","LSTM_bit" + ".h5"))
model.save(os.path.join("DATA","LSTM_gold" + ".h5"))


# 预测
# 预测天数
days=15
# 从第几天开始
start_day=30
test=dataset[start_day-timestep:start_day]
testPredict=predict_future(days,test,model).tolist()
# testPredict = scaler.inverse_transform(testPredict).tolist()
testPredict=np.reshape(df.iloc[:start_day,1].values,(start_day,1)).tolist()+testPredict
date=df.iloc[:start_day+days,0].values.tolist()
real_value=df.iloc[:start_day+days,1].values.tolist()
dic={'Date':np.reshape(date,(len(date))),'predict value':np.reshape(testPredict,(len(testPredict))),
'real value':np.reshape(real_value,(len(real_value))) }
# 改
pd.DataFrame(dic).to_csv('../result/黄金预测15天趋势.csv',index=False)
# pd.DataFrame(dic).to_csv('../result/比特币预测15天趋势.csv',index=False)



# #获得每一天的预测结果
# testPredict = model.predict(testX)
# testPredict = scaler.inverse_transform(testPredict).tolist()
# value=dataset[:len(dataset)-len(testPredict)]
# value=scaler.inverse_transform(value).tolist()
# value=value+testPredict

# date=df.iloc[:len(value),0].values.tolist()

# real_value=scaler.inverse_transform(dataset).tolist()

# dic={'Date':np.reshape(date,(len(date))),'predict value':np.reshape(value,(len(value))),
# 'real value':np.reshape(real_value,(len(real_value))) }
# # 改
# pd.DataFrame(dic).to_csv('../result/黄金趋势(30天后).csv',index=False)
# # pd.DataFrame(dic).to_csv('../result/比特币趋势(30天后).csv',index=False)


# var模型获得p2
data = df.iloc[:,1].values
# 将整型变为float
data = data.astype('float32')
p2=VaR(data,alpha=0.5)/data[len(data)-1]*days




# 投资
