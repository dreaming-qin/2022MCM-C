from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
plt.rcParams['font.sans-serif']=['simhei']#用于正常显示中文标签
plt.rcParams['axes.unicode_minus']=False#用于正常显示负号



data = pd.read_csv('../BCHAIN-MKPRU.csv', encoding='utf-8', index_col='Date')
data.index = pd.to_datetime(data.index)  # 将字符串索引转换成时间索引

# data.plot(figsize=(12,8),marker='o',color='black',ylabel='客运量')#画图
# plt.show()

# sm.tsa.adfuller(data,regression='c')
# sm.tsa.adfuller(data,regression='nc')
# sm.tsa.adfuller(data,regression='ct')

diff=data.diff(1)
diff.dropna(inplace=True)
diff.plot(figsize=(12,8),marker='o',color='black')#画图
# plt.show()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(diff.values.squeeze(), lags=12, ax=ax1)#自相关系数图1阶截尾,决定MA（1）
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(diff, lags=12, ax=ax2)#偏相关系数图1阶截尾,决定AR（1）

model = ARIMA(data, order=(1, 1, 1)).fit()#拟合模型
model.summary()#统计信息汇总

#系数检验
params=model.params#系数
tvalues=model.tvalues#系数t值
bse=model.bse#系数标准误
pvalues=model.pvalues#系数p值

#绘制残差序列折线图
resid=model.resid#残差序列
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax = model.resid.plot(ax=ax)

#计算模型拟合值
# fit=model.predict(exog=data[['TLHYL']])

#8.1.检验序列自相关
sm.stats.durbin_watson(model.resid.values)#DW检验：靠近2——正常；靠近0——正自相关；靠近4——负自相关

#8.2.AIC和BIC准则
model.aic#模型的AIC值
model.bic#模型的BIC值

#8.3.残差序列正态性检验
stats.normaltest(resid)#检验序列残差是否为正态分布
#最终检验结果显示无法拒绝原假设，说明残差序列为正态分布，模型拟合良好

#8.4.绘制残差序列自相关图和偏自相关图
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=12, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=12, ax=ax2)
#如果两图都零阶截尾，这说明模型拟合良好


#预测至2016年的数据。由于ARIMA模型有两个参数，至少需要包含两个初始数据，因此从2006年开始预测
predict = model.predict('2016', '2022', dynamic=True)
print(predict)

#画预测图及置信区间图
fig, ax = plt.subplots(figsize=(10,8))
fig = plot_predict(model, start='2016', end='2022', ax=ax)
legend = ax.legend(loc='upper left')



