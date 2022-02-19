
from problem1_al import VaR,  format_date, get_result, investment,predict_future
import numpy as np
import  pandas as pd
import  os
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler


for aaa in range(11):
    alpha=aaa/10
    # if __name__ =='__main__':
    # 投资，得到利润，还有买入卖出的时间点
    # 以30天为一个周期，从第30天开始
    time_step=30
    # 预测15天
    predict_days=15
    # 在第三种状态
    is_gold_buy,is_bit_buy=False,False
    # 保证相同时间，一般来说，是黄金比比特币快，需要比特币跟上
    gold_p,bit_p=30,30
    z_gold,z_bit=0.0204,0.0413
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

    # 准备工作完成，开始计算fg,fp
    # 超参数alpha
    alpha=0.6
    temp=1000
    while bit_p<df_bit.shape[0] and gold_p<df_gold.shape[0]:
        test_LSTM_bit,test_VaR_bit=data_LSTM_bit[bit_p-30:bit_p],data_VaR_bit[bit_p-30:bit_p]
        var1=gold_p if gold_p<31 else 31
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
            f_bit=get_result(predict_days,test_LSTM_bit,test_VaR_bit,model_bit,scaler_bit,z_bit,alpha)
            f_gold=get_result(predict_days,test_LSTM_gold,test_VaR_gold,model_gold,scaler_gold,z_gold,alpha)

            if f_bit-f_gold>0 and f_bit>0:
                is_bit_buy,is_gold_buy,status=investment(1,buy_list,status,test_VaR_gold[len(test_VaR_gold)-1],df_gold.iloc[gold_p,0])
                is_bit_buy,is_gold_buy,status=investment(2,buy_list,status,test_VaR_bit[len(test_VaR_bit)-1],df_bit.iloc[bit_p,0])
            elif f_gold<0:
                is_bit_buy,is_gold_buy,status=investment(1,buy_list,status,test_VaR_gold[len(test_VaR_gold)-1],df_gold.iloc[gold_p,0])
            bit_p+=1
            gold_p+=1
            continue

        if (not is_gold_buy) and is_bit_buy:
            f_bit=get_result(predict_days,test_LSTM_bit,test_VaR_bit,model_bit,scaler_bit,z_bit,alpha)
            if df_bit.iloc[bit_p,0]!=df_gold.iloc[gold_p,0]:
                if f_bit<0:
                    is_bit_buy,is_gold_buy,status=investment(3,buy_list,status,test_VaR_bit[len(test_VaR_bit)-1],df_bit.iloc[bit_p,0])
                bit_p+=1
                continue
            # 获得分数f
            f_gold=get_result(predict_days,test_LSTM_gold,test_VaR_gold,model_gold,scaler_gold,z_gold,alpha)

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
            f_bit=get_result(predict_days,test_LSTM_bit,test_VaR_bit,model_bit,scaler_bit,z_bit,alpha)

            if df_bit.iloc[bit_p,0]!=df_gold.iloc[gold_p,0]:
                if f_bit>0:
                    is_bit_buy,is_gold_buy,status=investment(2,buy_list,status,test_VaR_bit[len(test_VaR_bit)-1],df_bit.iloc[bit_p,0])
                bit_p+=1
                continue
            # 获得分数f
            f_gold=get_result(predict_days,test_LSTM_gold,test_VaR_gold,model_gold,scaler_gold,z_gold,alpha)

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

    print('最终比例是{}'.format(status))
    dic={'record':np.reshape(buy_list,(len(buy_list)))}
    pd.DataFrame(dic).to_csv('../result/交易记录(beta={})dong.csv'.format(aaa/10),index=False)

    dic={'money':np.reshape(money,(len(money))),'date':np.reshape(date,(len(date)))}
    pd.DataFrame(dic).to_csv('../result/资产(beta={})dong.csv'.format(aaa/10),index=False)

