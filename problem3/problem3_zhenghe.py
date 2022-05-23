
import  pandas as pd


# 用来整合交易记录的
li=[-0.005,0,0.005,0.01]
for z in li:
    df_money = pd.read_csv('../result/problem3/利润(z={}).csv'.format(z), engine='python')
    df_trade=pd.read_csv('../result/problem3/交易记录(z={}).csv'.format(z), engine='python')
    data_trade=df_trade.iloc[:,0].values
    data_money=df_money.iloc[:,1].values
    data_date=df_money.iloc[:,0].values
    data_buy=['' for i in range(len(data_date))]
    p_money,p_trade=0,0
    while p_money<len(data_money):
        data_trade[p_trade]=data_trade[p_trade].replace('-','/')
        var1=data_date[p_money].split('/')
        if len(var1[1])<2:
            var1[1]='0'+var1[1]
        if len(var1[2])<2:
            var1[2]='0'+var1[2]
        data_date[p_money]='{}/{}/{}'.format(var1[0],var1[1],var1[2])
        if data_date[p_money] in data_trade[p_trade]:
            index = data_trade[p_trade].find(' ') #第一次出现的位置
            index2=data_trade[p_trade].find(' ',index+1) #第二次出现的位置
            data_buy[p_money]=data_trade[p_trade][:index2]
            p_trade+=1
        p_money+=1

    dic={'date':data_date,'money':data_money,'trade':data_buy}
    pd.DataFrame(dic).to_csv('../result/problem3/交易(z={}).csv'.format(z),index=False)