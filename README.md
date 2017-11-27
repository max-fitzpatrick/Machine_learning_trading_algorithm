# Machine_learning_trading_algorithm
Master's degree project: Development of a trading algorithm which uses supervised machine learning classification techniques to generate buy/sell signals

ABSTRACT

This research paper attempts to study the usefulness of popular technical indicators for predicting future movements in S&P 500 equities. We train a unique random forest classifier on daily technical indicators derived from OHLC data from 01/01/2000 to 19/09/2014 for 75 S&P 500 equities and attempt to predict whether the stock will be up or down in price 30 trading days later. The classifier is incorporated into an algorithmic trading strategy which uses the up/down prediction as a buy/sell signal for the underlying security. Several strategies were created and tested, ranging from single equity strategies to directional long/short investment strategies. 

The out of sample performance (01/01/2015 to 01/01/2017) is extremely promising and culminated in two long/short algorithms which beat the benchmark over the tested time-period while maintaining better risk metrics than a SPY buy and hold strategy. The profitability of these strategies simultaneously indicates that technical indicators do indeed have useful predictive power for future price movements, that a relevant feature set in conjunction with machine learning classification techniques can profitably predict price movements in stock markets, and that identifiable recurring patterns and trends exist in financial markets. 
