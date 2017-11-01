#Diversified long/short trading algorithm. A long or short position is taken for each stock in our universe based on its classifier's output.
#Topic: Developing an algorithmic trading strategy using supervised machine learning classification techniques and technical indicators
#Student: Max Fitzpatrick
#Professor: Yves Jegourel
#Course: MSc Banking, Finance & Commodity Trading
#University: University of Bordeaux
################################################################
################################################################
import quantopian.optimize as opt
import numpy as np
import pandas as pd
import math as m
import datetime
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

#Pulling clf training data from csv (Quantopian workaround)
def grab_data(df):
    global my_data
    my_data = df
    return df

#Training classifier functions (Quantopian workaround to avoid timing out)
def train_classifiers(context):
    training_Data = pd.DataFrame(np.random.randint(0,100,size=(len(my_data), 2)), columns=list('AB'))
    training_Data['Delta'] = my_data.ix[:, context.training_counter]
    training_Data['MA10'] = my_data.ix[:,context.training_counter+1]
    training_Data['MA20'] = my_data.ix[:,context.training_counter+2]
    training_Data['MA100'] = my_data.ix[:,context.training_counter+3]
    training_Data['ROC'] = my_data.ix[:,context.training_counter+4]
    training_Data['B1'] = my_data.ix[:,context.training_counter+5]
    training_Data['B2'] = my_data.ix[:,context.training_counter+6]
    training_Data['SOk'] = my_data.ix[:,context.training_counter+7]
    training_Data['MACD'] = my_data.ix[:,context.training_counter+8]
    training_Data['MACDsign'] = my_data.ix[:,context.training_counter+9]
    training_Data['MACDdiff'] = my_data.ix[:,context.training_counter+10]
    training_Data['RSI14'] = my_data.ix[:,context.training_counter+11]
    training_Data['CCI'] = my_data.ix[:,context.training_counter+12]
    training_Data['Class'] = my_data.ix[:,context.training_counter+13]
    training_Data = training_Data.drop(['A', 'B'], axis=1)
    training_Data = training_Data.head(3400)
    context.training_counter += 14
    X = training_Data.drop(['Class'], 1)
    y = training_Data['Class']
    X = np.array(X)
    y = np.array(y)
    clf = RandomForestClassifier(random_state=20, n_estimators=1000)
    clf.fit(X, y)
    clf_list[ticker_list[context.asset_counter]] = clf
    print("Random forest classifier is trained and stored for :", ticker_list[context.asset_counter])
    context.asset_counter += 1
    
    return clf_list, context

########################
#####MAIN FUNCTIONS#####
########################

#Initialize sets our global variables
def initialize(context):
    #Defining our universe
    set_symbol_lookup_date('2015-01-01')
    context.assets = symbols("aapl",
                             "abt",
                             "adm",
                             "adp",
                             "adsk",
                             "aet",
                             "aig",
                             "amgn",
                             "axp",
                             "bac",
                             "bax",
                             "bby",
                             "bdx",
                             "bll",
                             "ca",
                             "cag",
                             "ci",
                             "clx",
                             "csco",
                             "csx",
                             "dis",
                             "dov",
                             "ecl",
                             "emn",
                             "emr",
                             "fdx",
                             "gis",
                             "gpc",
                             "gps",
                             "gww",
                             "hon",
                             "hrb",
                             "intc",
                             "itw",
                             "jnj",
                             "jwn",
                             "key",
                             "lb",
                             "lly",
                             "lmt",
                             "lnc",
                             "low",
                             "luv",
                             "mat",
                             "mdt",
                             "mmc",
                             "mro",
                             "msft",
                             "mu",
                             "nem",
                             "nke",
                             "noc",
                             "ntap",
                             "nue",
                             "nwl",
                             "orcl",
                             "oxy",
                             "pcar",
                             "ph",
                             "pki",
                             "px",
                             "shw",
                             "slb",
                             "sna",
                             "sti",
                             "swk",
                             "syy",
                             "tgt",
                             "tjx",
                             "txt",
                             "unh",
                             "wfc",
                             "wmb",
                             "wmt",
                             "xom")
    global ticker_list
    ticker_list = ["aapl",
                   "abt",
                   "adm",
                   "adp",
                   "adsk",
                   "aet",
                   "aig",
                   "amgn",
                   "axp",
                   "bac",
                   "bax",
                   "bby",
                   "bdx",
                   "bll",
                   "ca",
                   "cag",
                   "ci",
                   "clx",
                   "csco",
                   "csx",
                   "dis",
                   "dov",
                   "ecl",
                   "emn",
                   "emr",
                   "fdx",
                   "gis",
                   "gpc",
                   "gps",
                   "gww",
                   "hon",
                   "hrb",
                   "intc",
                   "itw",
                   "jnj",
                   "jwn",
                   "key",
                   "lb",
                   "lly",
                   "lmt",
                   "lnc",
                   "low",
                   "luv",
                   "mat",
                   "mdt",
                   "mmc",
                   "mro",
                   "msft",
                   "mu",
                   "nem",
                   "nke",
                   "noc",
                   "ntap",
                   "nue",
                   "nwl",
                   "orcl",
                   "oxy",
                   "pcar",
                   "ph",
                   "pki",
                   "px",
                   "shw",
                   "slb",
                   "sna",
                   "sti",
                   "swk",
                   "syy",
                   "tgt",
                   "tjx",
                   "txt",
                   "unh",
                   "wfc",
                   "wmb",
                   "wmt",
                   "xom"]
                    
    #Declaring global variables
    context.h = 30 # <- investment time horizon is 30 trading days
    context.training_counter = 1
    context.asset_counter = 0
    global clf
    global clf_list
    clf_list = {}
    context.training_check = 0
    context.trained = False
    
    #Pulling our training data from dropbox
    fetch_csv("https://dl.dropboxusercontent.com/s/xih9k23niw8j06r/MEGALS75.csv", pre_func=grab_data, date_column='Date', symbol='aapl')
        
    #Setting our portfolio to rebalance every 30 trading days
    context.iDays = 30
    context.NDays = 30  # <- Set to desired N days market is open
    schedule_function(func=rebalance, 
                      date_rule=date_rules.every_day(),
                      half_days=True,
                      time_rule=time_rules.market_open(hours=0,minutes=1))

#We use handle data, which is called once per minute, in order to train our classifiers. Doing each classifier one by one avoids timing out on Quantopian's platform
def handle_data(context, data):
    if context.training_check == 0:
        train_classifiers(context)
        if context.asset_counter == len(ticker_list):
            context.training_check = 1
            context.trained = True
        else:
            return
        
    else:
        pass

#rebalance function that runs every 30 trading days
def rebalance(context,data):
    if context.trained == False:
        return
    context.iDays += 1
    if (context.iDays % context.NDays) != 1:
        return
    # Start rebalance
    log.info("starting rebalance")    
    #Generating new trades for the next 30 day period
    i = 0
    targetlist = pd.DataFrame(np.random.randint(0,100,size=(len(ticker_list), 3)))
    targetlist.columns = ['Ticker', 'LongProbability', 'Class']
    #Pulling OHLC data for the past 100 days and then generating our technical indicators for each asset in our universe
    for asset in context.assets:
        try:
            df = pd.DataFrame(data.history(asset,'open', 100, '1d'))
            df.columns = ['Adj. Open']
            df['Adj. High'] = data.history(asset,'high', 100, '1d')
            df['Adj. Low'] = data.history(asset,'low', 100, '1d')
            df['Adj. Close'] = data.history(asset,'close', 100, '1d')

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

            X_recent = np.array(context.latest_Indicators)
            LongProbability = clf_list[ticker_list[i]].predict_proba(X_recent)
            LongProbability = LongProbability[:,1]

            targetlist.ix[i,'Ticker'] = ticker_list[i]
            targetlist.ix[i,'LongProbability'] = LongProbability
            if targetlist.ix[i,'LongProbability'] >= 0.5:
                targetlist.ix[i,'Class'] = 1
            else:
                targetlist.ix[i,'Class'] = -1
            i += 1
        
        except:
            print("ERROR FOR: ", asset)
            targetlist.ix[i,'Ticker'] = ticker_list[i]
            targetlist.ix[i,'LongProbability'] = 0.5
            targetlist.ix[i,'Class'] = 1
            i += 1
    
    #Creating shortlists of our long and short positions based on classifier output
    targetlist['LongProbability'] = targetlist['LongProbability'].astype('float')
    targetlist.sort_values(by = ['LongProbability'], ascending = False, inplace = True)
    print(targetlist)
    long_counter = 0
    short_counter = 0
    
    for i in range(len(targetlist)):
        if targetlist.ix[i,'LongProbability'] > 0.5:
            long_counter +=1
           
        if targetlist.ix[i,'LongProbability'] < 0.5:
            short_counter +=1
    
    longs = targetlist.head(long_counter)
    shorts = targetlist.tail(short_counter)
    print(longs)
    print(shorts)
    
    try:
        weight = float(len(longs)+len(shorts))
        weight = float(1/weight)
        print("WEIGHT: ", weight)
        long_weight = float(weight)
        short_weight = float(-weight)
    except:
        weight = 0
        print("WEIGHT: ", weight)
        long_weight = 0
        short_weight = 0
    
    long_list = []
    short_list = []
    
    for i in range(len(ticker_list)):
        if ticker_list[i] in longs['Ticker'].values:
            long_list.append(context.assets[i])
            
    for i in range(len(ticker_list)):
        if ticker_list[i] in shorts['Ticker'].values:
            short_list.append(context.assets[i])        
    
    weights = {long_list[i]: long_weight for i in range(len(long_list))}  
    weights.update({short_list[i]: short_weight for i in range(len(short_list))})
    print(weights)
    
    #Placing the relevant orders
    order_optimal_portfolio(objective=opt.TargetWeights(weights), constraints=[opt.MaxGrossExposure(1.0)])
    
    record(leverage = context.account.leverage)
