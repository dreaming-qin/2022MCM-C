import numpy as np
import math

from problem1_al import VaR, predict_future

# 下行风险，输入是每天的价格data，n*1的array；每天的资本，n*1的array
# 输出是下行风险值
def downside_risk(data,money):
    data_sub=[data[i+1]-data[i] for i in range(len(data)-1)]
    data_mean=np.mean(data_sub)
    ans=0
    index=0
    for i in range(len(money)):
        if money[i]<data_mean:
            ans+=(money[i]-data_mean)**2
            index+=1
    return math.sqrt(ans/index)


# 最大回撤，输入是资金序列，形状是n*1的array，滑动窗口大小和滑动窗口移动速度
# 输出是平均回撤值，个n*1的list
def maximum_drawdown(data,windown_size=100,step=15):
    index=int((len(data)-windown_size)/step)+1
    drawdown=[0 for i in range(index)]
    point=[0,int(windown_size/2),windown_size]
    index=0
    flag=True
    while flag:
        if point[2]>len(data):
            point=[len(data)-windown_size,int((2*len(data)-windown_size)/2),len(data)]
            flag=False
        max_=max(data[point[0]:point[1]])
        min_=min(data[point[1]:point[2]])
        drawdown[index]=(max_-min_)/max_
        point+=15
    return drawdown
    
# 累乘式的三次指数平滑
def predict_value_with_exp_smoothing_3(data,params=[0.2,0.05,0.9],T=30,predict_step=15):
    a,r,b=params[0],params[1],params[2]
    # 初始化S,B,I
    S=[data[i] if i<=T else 0 for i in range(len(data))]
    B=[data[i+1]-data[i] if i<=T else 0 for i in range(len(data))]
    I=[1 if i<=T else 0 for i in range(len(data))]
    for i in range(T+1,len(data)):
        S[i]=a*(data[i]/I[i-T])+(1-a)*(S[i-1]+B[i-1])
        B[i]=r*(S[i]-S[i-1])+(1-r)*B[i-1]
        I[i]=b*(data[i]/S[i])+(1-b)*I[i-T]
    len_data=len(data)-1
    ans=[0 for i in range(predict_step)]
    for i in range(predict_step):
        ans[i]=(S[len_data]+i*B[len_data])*I[len_data-T+(i%T)]
    return ans

def predict_by_LSTM(LSTM_data,VaR_data,model,scaler,predict_days=15):
    p1_bit=predict_future()(predict_days,LSTM_data,model).tolist()
    p1_bit = scaler.inverse_transform(p1_bit).tolist()
    p1_bit=p1_bit[predict_days-1]/VaR_data[len(VaR_data)-1]-1
    return p1_bit

def predict_by_VaR(VaR_data,predict_days=15):
    p2_bit=VaR(VaR_data,c=0.8)/VaR_data[len(VaR_data)-1]*predict_days
    return p2_bit
