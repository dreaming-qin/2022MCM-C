
from problem1_al import VaR,  format_date, get_result, investment,predict_future
import numpy as np
import  pandas as pd
import  os
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler

from problem2_al import downside_risk, get_result_by_state, maximum_drawdown, predict_value_with_exp_smoothing_3

# 用来获得数据的
li=[0]
state=0
for z in li:
    # 投资，得到利润，还有买入卖出的时间点
    # 以30天为一个周期，从第30天开始
    time_step=30
    # 预测15天
    predict_days=15
    # 在第三种状态
    is_gold_buy,is_bit_buy=False,False
    # 保证相同时间，一般来说，是黄金比比特币快，需要比特币跟上
    gold_p,bit_p=30,30
    if state==3:
        gold_p,bit_p=70,70
    z_gold,z_bit=0.0204+z,0.0413+z
    print(z_gold,z_bit)
    # 模型
    model_gold = load_model(os.path.join("DATA","LSTM_bit" + ".h5"))
    model_bit = load_model(os.path.join("DATA","LSTM_gold" + ".h5"))


    # 三元组状态，现金，黄金，比特币
    status=[1000,0,0]
    # 交易列表，包含时间，买卖，物品
    buy_list=[]
    # 资产
    money=[]
    date=[]

    df_bit = pd.read_csv('../BCHAIN-MKPRU.csv', engine='python', skipfooter=3)
    df_gold = pd.read_csv('../LBMA-GOLD.csv', engine='python', skipfooter=3)
    df_bit,df_gold=format_date(df_bit),format_date(df_gold)

    # LSTM获得数据集
    data_LSTM_gold,data_LSTM_bit = df_gold.iloc[:,1].values,df_bit.iloc[:,1].values
    data_LSTM_gold=np.reshape(data_LSTM_gold,(len(data_LSTM_gold),-1))
    data_LSTM_bit=np.reshape(data_LSTM_bit,(len(data_LSTM_bit),-1))
    # 将整型变为float
    data_LSTM_gold,data_LSTM_bit = data_LSTM_gold.astype('float32'),data_LSTM_bit.astype('float32')
    #归一化
    scaler_bit = MinMaxScaler(feature_range=(0, 1))
    scaler_gold=MinMaxScaler(feature_range=(0, 1))
    data_LSTM_gold,data_LSTM_bit = scaler_gold.fit_transform(data_LSTM_gold),scaler_bit.fit_transform(data_LSTM_bit)

    # VaR获得数据集
    data_VaR_gold,data_VaR_bit = df_gold.iloc[:,1].values,df_bit.iloc[:,1].values
    data_VaR_gold,data_VaR_bit = data_VaR_gold.astype('float32'),data_VaR_bit.astype('float32')

    #让alpha从0到1的模板 

    # 准备工作完成，开始计算fg,fp
    # 超参数alpha
    alpha=0.6
    temp=1000
    while bit_p<df_bit.shape[0] and gold_p<df_gold.shape[0]:
        test_LSTM_bit,test_VaR_bit=data_LSTM_bit[bit_p-time_step:bit_p],data_VaR_bit[bit_p-time_step:bit_p]
        var1=gold_p if gold_p<101 else 101
        test_LSTM_gold,test_VaR_gold=data_LSTM_gold[gold_p-var1:gold_p],data_VaR_gold[gold_p-var1:gold_p]

        # 计算当前资产
        var1=status[0]+0.99*status[1]*test_VaR_gold[len(test_VaR_gold)-1]+0.98*status[2]*test_VaR_bit[len(test_VaR_bit)-1]
        money.append(var1)
        if abs(var1-temp)>=10000:
            alpha+=-0.05 if var1>temp else 0.05
            temp=var1
        # 日期
        date.append(df_bit.iloc[bit_p,0])

        if is_gold_buy and  (not is_bit_buy):
            if df_bit.iloc[bit_p,0]!=df_gold.iloc[gold_p,0]:
                bit_p+=1
                continue
            # 获得分数f
            f_bit=get_result_by_state(state,df_bit,bit_p, predict_days,test_LSTM_bit,test_VaR_bit,model_bit,scaler_bit,z_bit,alpha)
            f_gold=get_result_by_state(state,df_gold,gold_p,predict_days,test_LSTM_gold,test_VaR_gold,model_gold,scaler_gold,z_gold,alpha)

            if f_bit-f_gold>0 and f_bit>0:
                is_bit_buy,is_gold_buy,status=investment(1,buy_list,status,test_VaR_gold[len(test_VaR_gold)-1],df_gold.iloc[gold_p,0])
                is_bit_buy,is_gold_buy,status=investment(2,buy_list,status,test_VaR_bit[len(test_VaR_bit)-1],df_bit.iloc[bit_p,0])
            elif f_gold<0:
                is_bit_buy,is_gold_buy,status=investment(1,buy_list,status,test_VaR_gold[len(test_VaR_gold)-1],df_gold.iloc[gold_p,0])
            bit_p+=1
            gold_p+=1
            continue

        if (not is_gold_buy) and is_bit_buy:
            f_bit=get_result_by_state(state,df_bit,bit_p,predict_days,test_LSTM_bit,test_VaR_bit,model_bit,scaler_bit,z_bit,alpha)
            if df_bit.iloc[bit_p,0]!=df_gold.iloc[gold_p,0]:
                if f_bit<0:
                    is_bit_buy,is_gold_buy,status=investment(3,buy_list,status,test_VaR_bit[len(test_VaR_bit)-1],df_bit.iloc[bit_p,0])
                bit_p+=1
                continue
            # 获得分数f
            f_gold=get_result_by_state(state,df_gold,gold_p,predict_days,test_LSTM_gold,test_VaR_gold,model_gold,scaler_gold,z_gold,alpha)

            is_buy_gold=(status[0]+status[2]*test_VaR_bit[len(test_VaR_bit)-1]*0.98)*0.99>test_VaR_gold[len(test_VaR_gold)-1]
            if f_gold-f_bit>0 and is_buy_gold and f_gold>0:            
                is_bit_buy,is_gold_buy,status=investment(3,buy_list,status,test_VaR_bit[len(test_VaR_bit)-1],df_bit.iloc[bit_p,0])
                is_bit_buy,is_gold_buy,status=investment(0,buy_list,status,test_VaR_gold[len(test_VaR_gold)-1],df_gold.iloc[gold_p,0])
            elif f_bit<0:
                is_bit_buy,is_gold_buy,status=investment(3,buy_list,status,test_VaR_bit[len(test_VaR_bit)-1],df_bit.iloc[bit_p,0])
            bit_p+=1
            gold_p+=1
            continue

        if (not is_gold_buy) and (not is_bit_buy):
            f_bit=get_result_by_state(state,df_bit,bit_p,predict_days,test_LSTM_bit,test_VaR_bit,model_bit,scaler_bit,z_bit,alpha)

            if df_bit.iloc[bit_p,0]!=df_gold.iloc[gold_p,0]:
                if f_bit>0:
                    is_bit_buy,is_gold_buy,status=investment(2,buy_list,status,test_VaR_bit[len(test_VaR_bit)-1],df_bit.iloc[bit_p,0])
                bit_p+=1
                continue
            # 获得分数f
            f_gold=get_result_by_state(state,df_gold,gold_p,predict_days,test_LSTM_gold,test_VaR_gold,model_gold,scaler_gold,z_gold,alpha)

            is_buy_gold=status[0]*0.99>test_VaR_gold[len(test_VaR_gold)-1]
            if f_bit<0 and f_gold<0:
                bit_p+=1
                gold_p+=1
                continue

            if f_gold>f_bit and is_buy_gold:
                is_bit_buy,is_gold_buy,status=investment(0,buy_list,status,test_VaR_gold[len(test_VaR_gold)-1],df_gold.iloc[gold_p,0])
            elif f_bit>f_gold:
                is_bit_buy,is_gold_buy,status=investment(2,buy_list,status,test_VaR_bit[len(test_VaR_bit)-1],df_bit.iloc[bit_p,0])
            bit_p+=1
            gold_p+=1
            continue
    # 先算回撤值
    drawdown=maximum_drawdown(np.array(money))

    # 然后算下行风险
    downside_r=downside_risk(df_bit.iloc[:,1].values,
                                df_gold.iloc[:,1].values,
                                np.array(money))

    dic={'value':np.reshape(buy_list,(len(buy_list)))}
    pd.DataFrame(dic).to_csv('../result/交易记录(z={}).csv'.format(z),index=False)

    dic={'value':np.reshape(drawdown,(len(drawdown)))}
    pd.DataFrame(dic).to_csv('../result/回撤(z={}).csv'.format(z),index=False)
    dic={'value':np.reshape(money,(len(money)))}
    pd.DataFrame(dic).to_csv('../result/利润(z={}).csv'.format(z),index=False)
    dic={'value':[downside_r]}
    pd.DataFrame(dic).to_csv('../result/下行风险(z={}).csv'.format(z),index=False)






# # 用来整合交易记录的
# li=[-0.01,-0.005,0,0.005,0.01]
# for z in li:
#     df_money = pd.read_csv('../result/problem3/利润(z={}).csv'.format(z), engine='python')
#     df_trade=pd.read_csv('../result/problem3/交易记录(z={}).csv'.format(z), engine='python')
#     data_trade=df_trade.iloc[:,0].values
#     data_money=df_money.iloc[:,1].values
#     data_date=df_money.iloc[:,0].values
#     data_buy=['' for i in range(len(data_date))]
#     p_money,p_trade=0,0
#     while p_money<len(data_money):
#         data_trade[p_trade]=data_trade[p_trade].replace('-','/')
#         if data_date[p_money] in data_trade[p_trade]:
#             index = data_trade[p_trade].find(' ') #第一次出现的位置
#             index2=data_trade[p_trade].find(' ',index+1) #第二次出现的位置
#             data_buy[p_money]=data_trade[p_trade][:index2]
#             p_trade+=1
#         p_money+=1

#     dic={'date':data_date,'money':data_money,'trade':data_buy}
#     pd.DataFrame(dic).to_csv('../result/problem3/交易(z={}).csv'.format(z),index=False)

