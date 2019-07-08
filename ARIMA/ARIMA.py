# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa import arima_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings("ignore")


def ARIMA(series,n):
    '''
    只讨论一阶差分的ARIMA模型，预测，数字索引从1开始
    series:时间序列
    n:需要往后预测的个数
    '''
    series = np.array(series)
    series = pd.Series(series.reshape(-1))
    currentDir = os.getcwd()#当前工作路径
    #一阶差分数据
    fd = series.diff(1)[1:]
    plot_acf(fd).savefig(currentDir+'/一阶差分自相关图.png')
    plot_pacf(fd).savefig(currentDir+'/一阶差分偏自相关图.png')
    #一阶差分单位根检验
    unitP = adfuller(fd)[1]
    if unitP>0.05:
        unitAssess = '单位根检验中p值为%.2f，大于0.05，认为该一阶差分序列判断为非平稳序列'%(unitP)
        #print('单位根检验中p值为%.2f，大于0.05，认为该一阶差分序列判断为非平稳序列'%(unitP))
    else:
        unitAssess = '单位根检验中p值为%.2f，小于0.05，认为该一阶差分序列判断为平稳序列'%(unitP)
        #print('单位根检验中p值为%.2f，小于0.05，认为该一阶差分序列判断为平稳序列'%(unitP))
    #白噪声检验
    noiseP = acorr_ljungbox(fd, lags=1)[-1]
    if noiseP<=0.05:
        noiseAssess = '白噪声检验中p值为%.2f，小于0.05，认为该一阶差分序列为非白噪声'%noiseP
        #print('白噪声检验中p值为%.2f，小于0.05，认为该一阶差分序列为非白噪声'%noiseP)
    else:
        noiseAssess = '白噪声检验中%.2f，大于0.05，认为该一阶差分序列为白噪声'%noiseP
        #print('白噪声检验中%.2f，大于0.05，认为该一阶差分序列为白噪声'%noiseP)
    #BIC准则确定p、q值
    pMax = int(series.shape[0]/10)# 一般阶数不超过length/10
    qMax = pMax# 一般阶数不超过length/10
    bics = list()
    for p in range(pMax + 1):
        tmp = list()
        for q in range(qMax + 1):
            try:
                tmp.append(arima_model.ARIMA(series, (p, 1, q)).fit().bic)
            except Exception as e:
                #print(str(e))
                tmp.append(1e+10)#加入一个很大的数
        bics.append(tmp)
    bics = pd.DataFrame(bics)
    p, q = bics.stack().idxmin()
    #print('BIC准则下确定p,q为%s,%s'%(p,q))
    #建模
    model = arima_model.ARIMA(series,order=(p, 1, q)).fit()
    predict = model.forecast(n)[0]
    return {
            'model':{'value':model,'desc':'模型'},
            'unitP':{'value':unitP,'desc':unitAssess},
            'noiseP':{'value':noiseP[0],'desc':noiseAssess},
            'p':{'value':p,'desc':'AR模型阶数'},
            'q':{'value':q,'desc':'MA模型阶数'},
            'params':{'value':model.params,'desc':'模型系数'},
            'predict':{'value':predict,'desc':'往后预测%d个的序列'%(n)}
            }
    
if __name__ == "__main__":
    data = data = np.array([1.2,2.2,3.1,4.5,5.6,6.7,7.1,8.2,9.6,10.6,11,12.4,13.5,14.7,15.2])
    x = data[0:10]#输入数据
    y = data[10:]#需要预测的数据
    result = ARIMA(x,len(y))#预测结果,一阶差分偏自相关图,一阶差分自相关图
    predict = result['predict']['value']
    predict = np.round(predict,1)
    print('真实值:',y)
    print('预测值:',predict)
    print(result)
    
