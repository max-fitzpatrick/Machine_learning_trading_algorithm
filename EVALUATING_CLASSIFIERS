#Evaluating different machine learning classifiers
#Topic: Developing an algorithmic trading strategy using supervised machine learning classification techniques and technical indicators
#Student: Max Fitzpatrick
#Professor: Yves Jegourel
#Course: MSc Banking, Finance & Commodity Trading
#University: University of Bordeaux
################################################################
################################################################

#Importing the necessary libraries and packages
import numpy as np
import pandas as pd
import math as m
import quandl
import pickle
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn import preprocessing, cross_validation, neighbors, svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

################################################################
################################################################
#Building the necessary functions that we will use to generate our technical indicators
#Lags function
def buildLaggedFeatures(s,lag=2,dropna=True):
    if type(s) is pd.DataFrame:
        new_dict={}
        for col_name in s:
            new_dict[col_name]=s[col_name]
            # create lagged Series
            for l in range(1,lag+1):
                new_dict['%s_lag%d' %(col_name,l)]=s[col_name].shift(l)
        res=pd.DataFrame(new_dict,index=s.index)

    elif type(s) is pd.Series:
        the_range=range(lag+1)
        res=pd.concat([s.shift(i) for i in the_range],axis=1)
        res.columns=['lag_%d' %i for i in the_range]
    else:
        print("Only works for DataFrame or Series")
        return None
    if dropna:
        return res.dropna()
    else:
        return res 

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

################################################################
################################################################
#Pulling and cleaning data
#Defining investment time horizon
#Testing on Apple (AAPL) data
h = 30
start = "2000-01-01"
end = "2014-12-31"
auth_token = "YOUR_AUTHTOKEN_HERE"

#Pulling daily stock price
print("Time horizon is :", h)
df = quandl.get("WIKI/AAPL", authtoken=auth_token, start_date=start, end_date=end)
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
print("Stock data pulled")

#Calculating percent change since t-h
df['Delta'] = df['Adj. Close'].pct_change(h)

#Calculating a simple 10 day moving average
df['MA10'] = df['Adj. Close'].rolling(window=10).mean()

#Calculating a simple 20 day moving average
df['MA20'] = df['Adj. Close'].rolling(window=20).mean()

#Calculating a simple 100 day moving average
df['MA100'] = df['Adj. Close'].rolling(window=100).mean()

#Calculating rate of change over h periods
ROC(df,h)

#Calculating 20 day Bollinger bands
BBANDS(df,20)

#Calculating 14 day stochastic oscillator
STOK(df,14)

#Calculating MACD and MACD signal
MACD(df,12,26)

#Calculating RSI14
delta = df['Delta']
up, down = delta.copy(), delta.copy()
up[up < 0] = 0
down[down > 0] = 0
roll_up1 = pd.stats.moments.ewma(up, 14)
roll_down1 = pd.stats.moments.ewma(down.abs(), 14)
RS1 = roll_up1 / roll_down1
RSI1 = 100.0 - (100.0 / (1.0 + RS1))
df['RSI14'] = RSI1

#Calculating CCI20
CCI(df,20)

print("Technical indicators generated")
#Dropping unwanted variables
df = df.drop(['Adj. Open',  'Adj. High',  'Adj. Low', 'Adj. Close', 'Adj. Volume'], axis=1)

#Replacing all missing data (NaN) with -99999 which will be treated as an outlier by our ML algorithms
#Creating our class variable. 
#If in t+h the stock price moves up or stays the same it is assigned a value of 1, if the stock price moves down it is assigned a value of -1

df['Class'] = df['Delta'].apply(lambda x: 1 if x >= 0 else -1)
df['Class'] = df['Class'].shift(-h)
df = df[100:]
df = df[:-h]

df.fillna(-99999, inplace=True)
print(df.head(5))
print(df.tail(5))
print("n is equal to :", len(df))

################################################################
################################################################
#X is our feature set, y is our label
#Training on a random 67% subsample of the data, testing on a random 33% subsample of the data
X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33)
print("X and y defined")
#################################################################
#################################################################
#Defining and training our classifiers

clf1 = svm.SVC(kernel='linear')
print("Beginning training of SVC")
clf1.fit(X_train, y_train)
print("Testing accuracy")
accuracy1 = clf1.score(X_test, y_test)
print("Accuracy of SVC is :", accuracy1)

clf2 = neighbors.KNeighborsClassifier(n_jobs=-1, n_neighbors=5)
print("Beginning training of KNN")
clf2.fit(X_train, y_train)
print("Testing accuracy")
accuracy2 = clf2.score(X_test, y_test)
print("Accuracy of KNN is :", accuracy2)

clf3 = RandomForestClassifier(n_jobs=-1, random_state=20, n_estimators=1000, oob_score=True)
print("Beginning training of RF")
clf3.fit(X_train, y_train)
accuracy3 = clf3.score(X_test, y_test)
print("Accuracy of RF is: ", accuracy3)
print("Out-of-bag score estimate: ", clf3.oob_score_)

eclf = VotingClassifier(estimators=[('svc', clf1), ('knn', clf2), ('rf',clf3)], voting='hard')
eclf = eclf.fit(X_train, y_train)
print("Testing accuracy")
accuracy = eclf.score(X_test, y_test)
print("Accuracy of the voting classifer is :", accuracy)

##################################################
#The Random Forest classifier outperforms the SVC, KNN, and voting classifiers. We will be proceeding with the Random Forest classifier
#Analysing the performance of our Random Forest classifier
#Saving the clf as a pickle for later use
with open('TrainedRFClassifier.pickle','wb') as f:
    pickle.dump(clf3, f)

print("Feature importances: ", clf3.feature_importances_)
print("Classification probabilities: ", clf3.predict_proba(X_test))

#Creating a confusion matrix for our test data
predicted = clf3.predict(X_test)
y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(predicted, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print("CONFUSION MATRIX")
print(df_confusion)

#Displaying the actual class, predicted class, and probability score of our classifier
for i in range(len(y_test)):
    print("Actual outcome :: {} and Predicted outcome :: {} and Probability :: {}".format(list(y_test)[i], predicted[i], clf3.predict_proba(X_test)[i]))
