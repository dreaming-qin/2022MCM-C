from turtle import color
import matplotlib.pyplot as plt
import pandas as pd

def predict_value_with_exp_smoothing_3(data,params,T,predict_step):
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
    for i in range(1,predict_step+1):
        data.append((S[len_data]+i*B[len_data])*I[len_data-T+(i%T)])
    return data



df=pd.read_csv('../BCHAIN-MKPRU.csv')
dataset= df.iloc[:,1].tolist()

plt.figure()
plt.plot(dataset,'g-',label='dwell')
plt.legend(loc='best')
plt.savefig('1.png')

dataset=predict_value_with_exp_smoothing_3(data=dataset,params=[0.2,0.05,0.9],T=365,predict_step=100) 
plt.figure()
plt.plot(dataset,'g-',label='dwell',color='red')
plt.legend(loc='best')
plt.savefig('2.png')
plt.close()





