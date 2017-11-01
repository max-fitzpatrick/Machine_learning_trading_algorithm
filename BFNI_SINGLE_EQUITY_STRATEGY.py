#Single equity strategy. Example with XOM (Exxon mobil). Context.assets, set_benchmark, and the training data URL would need to be changed for a different stock. 
#Topic: Developing an algorithmic trading strategy using supervised machine learning classification techniques and technical indicators
#Student: Max Fitzpatrick
#Professor: Yves Jegourel
#Course: MSc Banking, Finance & Commodity Trading
#University: University of Bordeaux
################################################################
################################################################
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.filters.morningstar import Q1500US
import numpy as np
import pandas as pd
import math as m
import datetime
import quantopian.optimize as opt
from sklearn.ensemble import RandomForestClassifier

###########################
#####SUPPORT FUNCTIONS#####
###########################

#Defining the functions for generating our technical indicators
#Rate of Change  
def ROC(df, n):  
    M = df['Adj. Close'].diff(n - 1)  
    N = df['Adj. Close'].shift(n - 1)  
    ROC = pd.Series(M / N, name = 'ROC_' + str(n))  
    df['ROC'] = ROC  
    return df

#Bollinger Bands  
def BBANDS(df, n):  
    MA = pd.Series(pd.rolling_mean(df['Adj. Close'], n))  
    MSD = pd.Series(pd.rolling_std(df['Adj. Close'], n))  
    b1 = 4 * MSD / MA  
    B1 = pd.Series(b1, name = 'BollingerB_' + str(n))  
    df['B1'] = B1  
    b2 = (df['Adj. Close'] - MA + 2 * MSD) / (4 * MSD)  
    B2 = pd.Series(b2, name = 'Bollinger%b_' + str(n))  
    df['B2'] = B2 
    return df

#Stochastic oscillator %K  
def STOK(df,n):  
    SOk = pd.Series((df['Adj. Close'] - pd.rolling_min(df['Adj. Low'],n)) / (pd.rolling_max(df['Adj. High'],n) - pd.rolling_min(df['Adj. Low'],n)), name = 'SO%k')  
    df['SOk'] = SOk  
    return df

#MACD, MACD Signal and MACD difference  
def MACD(df, n_fast, n_slow):  
    EMAfast = pd.Series(pd.ewma(df['Adj. Close'], span = n_fast, min_periods = n_slow - 1))  
    EMAslow = pd.Series(pd.ewma(df['Adj. Close'], span = n_slow, min_periods = n_slow - 1))  
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(pd.ewma(MACD, span = 9, min_periods = 8), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))  
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))  
    df['MACD'] = MACD  
    df['MACDsign'] = MACDsign  
    df['MACDdiff'] = MACDdiff  
    return df

#Commodity Channel Index  
def CCI(df, n):  
    PP = (df['Adj. High'] + df['Adj. Low'] + df['Adj. Close']) / 3  
    CCI = pd.Series((PP - pd.rolling_mean(PP, n)) / pd.rolling_std(PP, n), name = 'CCI_' + str(n))  
    df['CCI'] = CCI  
    return df

#Pulling clf training data from csv (workaround needed for Quantopian)
def grab_data(df):
    global my_data
    my_data = df
    return df

########################
#####MAIN FUNCTIONS#####
########################

#Initialize sets our global variables
def initialize(context):
    context.assets = sid(8347)
    set_benchmark(symbol('XOM'))
    context.h = 30 # <- Investment time horizon
    context.latest_Indicators = []
    
    #Pulling in our training data from Google Drive
    fetch_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRal9Ha-CJxzQz2mPjffAMytOKWe4EZeD-EaxzJqDqPSapRCh1Wb9VkIwp17KAI1ngSDXv_Sg6rJF3X/pub?gid=813679663&single=true&output=csv', pre_func=grab_data, date_column='Date', symbol='aapl')
    
    #Moving that training data to a DataFrame that we can reference (Quantopian workaround)
    training_Data = pd.DataFrame(np.random.randint(0,100,size=(len(my_data), 2)), columns=list('AB'))
    training_Data['Delta'] = my_data['Delta']
    training_Data['MA10'] = my_data['MA10']
    training_Data['MA20'] = my_data['MA20']
    training_Data['MA100'] = my_data['MA100']
    training_Data['ROC'] = my_data['ROC']
    training_Data['B1'] = my_data['B1']
    training_Data['B2'] = my_data['B2']
    training_Data['SOk'] = my_data['SOk']
    training_Data['MACD'] = my_data['MACD']
    training_Data['MACDsign'] = my_data['MACDsign']
    training_Data['MACDdiff'] = my_data['MACDdiff']
    training_Data['RSI14'] = my_data['RSI14']
    training_Data['CCI'] = my_data['CCI']
    training_Data['Class'] = my_data['Class']
    training_Data = training_Data.drop(['A', 'B'], axis=1)
    print("N = ", len(training_Data))
    
    #Declaring our feautures and labels, training our classifier
    X = training_Data.drop(['Class'], 1)
    y = training_Data['Class']
    X = np.array(X)
    y = np.array(y)
    print("X and y defined")
    context.clf = RandomForestClassifier(n_jobs=-1, random_state=20, n_estimators=1000)
    context.clf.fit(X, y)
    print("Random forest classifier is trained")   
    
    #Scheduling our portfolio rebalance to take place every 30 trading days
    context.iDays = 0
    context.NDays = 30  # <- Set to desired N days market is open
    schedule_function(func=rebalance, 
                      date_rule=date_rules.every_day(),
                      half_days=True,
                      time_rule=time_rules.market_open(hours=0,minutes=30))



def handle_data(context, data):
    pass

#rebalance function that runs every 30 trading days
def rebalance(context,data):
    context.iDays += 1
    if (context.iDays % context.NDays) != 1:
        return
    #Start rebalance
    log.info("starting rebalance")    
    #Pulling OHLC data for the past 100 days and then generating our technical indicators
    df = pd.DataFrame(data.history(context.assets,'open', 100, '1d'))
    df.columns = ['Adj. Open']
    df['Adj. High'] = data.history(context.assets,'high', 100, '1d')
    df['Adj. Low'] = data.history(context.assets,'low', 100, '1d')
    df['Adj. Close'] = data.history(context.assets,'close', 100, '1d')

    context.latest_Indicators = pd.DataFrame(np.random.randint(0,100,size=(100, 2)), columns=list('AB'))
    context.latest_Indicators['Adj. Open'] = df['Adj. Open'].values
    context.latest_Indicators['Adj. High'] = df['Adj. High'].values
    context.latest_Indicators['Adj. Low'] = df['Adj. Low'].values
    context.latest_Indicators['Adj. Close'] = df['Adj. Close'].values
    context.latest_Indicators = context.latest_Indicators.drop(['A', 'B'], axis=1)
    context.latest_Indicators['Delta'] = context.latest_Indicators['Adj. Close'].pct_change(context.h)
    context.latest_Indicators['MA10'] = context.latest_Indicators['Adj. Close'].rolling(window=10).mean()
    context.latest_Indicators['MA20'] = context.latest_Indicators['Adj. Close'].rolling(window=20).mean()
    context.latest_Indicators['MA100'] = context.latest_Indicators['Adj. Close'].rolling(window=100).mean()
    ROC(context.latest_Indicators,context.h)
    BBANDS(context.latest_Indicators,20)
    STOK(context.latest_Indicators,14)
    MACD(context.latest_Indicators,12,26)
    
    delta = context.latest_Indicators['Delta']
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = pd.stats.moments.ewma(up, 14)
    roll_down1 = pd.stats.moments.ewma(down.abs(), 14)
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    context.latest_Indicators['RSI14'] = RSI1

    CCI(context.latest_Indicators,20)
    context.latest_Indicators = context.latest_Indicators.drop(['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close'], 1)
    
    context.latest_Indicators = context.latest_Indicators[-1:]
    print("FINAL VALUES")
    print(context.latest_Indicators)
    #Feeding yesterday's technical indicators into to our classifier in order to generate a prediction
    X_recent = np.array(context.latest_Indicators)
    context.y_hat = context.clf.predict(X_recent)

    LongProbability = context.clf.predict_proba(X_recent)
    LongProbability = LongProbability[:,1]
    print("probability of long is :", LongProbability)
    if LongProbability >= 0.5:
        weight = 1
    else:
        weight = -1
    #Taking the relevant long or short position, based on the classifier's output
    order_optimal_portfolio(objective=opt.TargetWeights({context.assets:weight}), constraints=[opt.MaxGrossExposure(1.0)])
    
    record(leverage = context.account.leverage)
