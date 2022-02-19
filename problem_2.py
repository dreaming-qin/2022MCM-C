from turtle import color
import matplotlib.pyplot as plt
import pandas as pd

from problem2_al import predict_value_with_exp_smoothing_3


df=pd.read_csv('../BCHAIN-MKPRU.csv')
dataset= df.iloc[:,1].tolist()
predict_by_smoothing3=predict_value_with_exp_smoothing_3(data=dataset,params=[0.2,0.05,0.9],T=365,predict_step=15) 