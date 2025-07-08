""""""
import pandas

"""Project 6 : Indicator Optimizer .  		  	   		 	 	 			  		 			     			  	 

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 

Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 

Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 

We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 

-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 

Student Name: Tarun Bhalla (replace with your name)  		  	   		 	 	 			  		 			     			  	 
GT User ID: tbhalla6 (replace with your User ID)  		  	   		 	 	 			  		 			     			  	 
GT ID: 904075828 (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data

# import TheoreticallyOptimalStrategy as tos
#
# #---- global variables ----- #
# symbol = 'JPM'
# sd = dt.datetime(2008, 1, 1)
# ed = dt.datetime(2009, 12, 31)
# data = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
# test_strategy = tos.testPolicy('JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)
# portvals = ms.compute_portvals(test_strategy, 100000)


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "tbhalla6"  # replace tb34 with your Georgia Tech username.


def bollinger_bands(prices, window_size):
    df = prices.copy()
    df = df.ffill().bfill()
    sma = df.rolling(window_size).mean()
    rstd = df.rolling(window=window_size).std()
    # print("Num of NAN" , df.isna().sum())
    symbol = prices.columns[0]
    print(symbol)
    # print(rm)
    df['upperBand'] = sma + (2 * rstd)
    df['lower_band'] = sma - (2 * rstd)
    band_val = (df[symbol] - df['lower_band']) / (df['upperBand'] - df['lower_band'])

    return band_val
    # df["band_value"] = band_value
    # df["upperBand"] = upperBand
    # df["lower_band"] = lower_band
    # # print(df)
    # plt.figure(figsize=(14, 7))
    # plt.plot(df["JPM"], label='Stock', color='r')
    # plt.plot(df["upperBand"], label='Upper Bound', color='b')
    # plt.plot(df["lower_band"], label='Lower Bound', color='g')
    # plt.title('Bollinger Bands')
    # plt.xlabel('Date')
    # plt.ylabel('Portfolio Value')
    # plt.legend()
    # # plt.show()
    # # plt.close()
    # plt.savefig('Bollinger Bands.png')


def get_ema(prices, window_size):
    df = prices.copy()

    ema = prices.ewm(span=window_size, adjust=False).mean()
    # df["ema"] = ema
    # plt.figure(figsize=(14, 7))
    # plt.plot(df["ema"], label='EMA of span = 20', color='b')
    # plt.plot(df["JPM"], label="JPM", color='pink')
    # plt.title('Exponential Moving Average')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.legend()
    # # plt.show()
    # # plt.close()
    # plt.savefig('EMA.png')
    return ema

def get_momentum(prices, window_size):
    df = prices
    # print(df)
    # df = prices/prices.iloc[0] ## Normalize prices
    symbol = prices.columns[0]

    momentum = prices[symbol].pct_change(window_size)
    # plt.figure(figsize=(14, 7))
    # plt.plot(df["Momentum"], label='Momentum with lookback = 20', color='b')
    # plt.plot(df["JPM"], label="JPM", color='pink')
    # plt.title('Momentum')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.legend()
    # # plt.show()
    # # plt.close()
    # plt.savefig('Momentum.png')

    return momentum

def get_rsi(prices, window_size):
    data = prices.copy()
    df = data.copy()
    symbol = prices.columns[0]

    delta = prices[symbol].diff()   ## getting price change of going up or down
    average_up = (delta.where(delta > 0, 0)).rolling(window=window_size).mean()
    average_down = (-delta.where(delta < 0, 0)).rolling(window=window_size).mean()
    # print(average_up, average_down)
    rs = average_up / average_down
    rsi = 100 - (100 / (1 + rs))
    # print(rsi)

    # df['rsi'] = rsi
    # plt.figure(figsize=(14, 7))
    # plt.plot(df["rsi"], label='RSI with window = 14', color='b')
    # plt.plot(df["JPM"], label="JPM", color='pink')
    # plt.title('RSI Indicator of 14 days')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.legend()
    # # plt.show()
    # # plt.close()
    # plt.savefig('RSI.png')
    return rsi


def get_cci(prices, window_size):
    data = prices.copy()
    df = data.copy()
    symbol = prices.columns[0]

    df['TP'] = (df[symbol])

    sma = df['TP'].rolling(20).mean()
    mad = df['TP'].rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())

    cci = (df['TP'] - sma) / (0.015 * mad)
    return cci # Remove leading NaNs




# if __name__ == "__main__":
#     # symbol = 'JPM'
    # sd = dt.datetime(2008, 1, 1)
    # ed = dt.datetime(2009, 12, 31)
    # data = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
    #
    # x = get_momentum(data, window_size=20)
    # print(x)