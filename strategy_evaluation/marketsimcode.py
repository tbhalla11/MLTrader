""""""
import pandas

"""MC2-P1: Market simulator.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
import os

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from util import get_data, plot_data

def testPolicy(symbol, sd, ed):

    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

    daily_returns = prices[symbol].diff()
    daily_returns = daily_returns.multiply(1000) # 1000 shares
    # print(daily_returns)

    ## building a DF for the trades and a counter for holdings
    df_trades = pd.DataFrame(index=prices.index, columns=[symbol], data=0)
    df_trades.rename(columns={symbol: 'Trades'}, inplace=True)
    shares = 0

    for index in range(prices.shape[0] - 1 ):
        # print(shares)
        #Scenario: Currently holding and stock goes down tomorrow(SELL)
        if daily_returns.iloc[index + 1] < 0 and shares >= 0:
            # print(daily_returns.iloc[index + 1], shares)
            shares -= 1000
            df_trades.iloc[index] = -1000

            # print(shares, daily_returns[index])
        #Scenario: Currently NOT holding and price goes UP tomorrow (BUY)
        if daily_returns.iloc[index + 1] > 0 and shares<=0:
            # print(daily_returns.iloc[index + 1], shares)

            df_trades.iloc[index] = 1000
            shares += 1000

    # print(df_trades)
    return df_trades

def compute_portvals(
    trades,
    start_val=100000,
    commission=9.95,
    impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    symbol = trades.columns[0]  # Get the first column name
    # print(trades.columns[0])

    # ------ Get order data (Reading from orders.csv) ----- #
    orders_df = trades
    orders_df = orders_df.sort_index()
    # print(orders_df.head())   ---- formating correct ---
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]
    if symbol != 'Trades':
        orders_df = orders_df.rename(columns={symbol: 'Trades'})
    # print(orders_df)
    orders_df.insert(0, 'Symbol', 'JPM')
    orders_df.insert(1, 'Order', 'SELL')  ### match format as prior marketsim
    # print(orders_df)

    for index in range(orders_df.shape[0]):
        if orders_df.iloc[index, 2] > 0:
            orders_df.iloc[index, 1] = 'BUY'
    # print(orders_df.head())
    orders_df['JPM'] = orders_df['Trades'].abs()  ### remove -1000 from trades

    # print(orders_df)
    ## get price history of symbols ----- from data
    dt_range = pd.date_range(start_date, end_date)
    prices = get_data(['JPM'], pd.date_range(start_date, end_date), addSPY=False)
    prices = prices[['JPM']].ffill().bfill()  # Ensure we only keep JPM column and fill NaNs
    prices['Cash'] = 1.0

    # print(prices)
    #----- Data for each Symbol produced ----

    # data fame for trades
    trades = pd.DataFrame(data=0.00, index=prices.index, columns=prices.columns)
    # print(trades)
    prices = prices.fillna(method='ffill')
    prices = prices.fillna(method='bfill')
    orders_df = orders_df.fillna(method='ffill')
    orders_df = orders_df.fillna(method='bfill')
    # print(orders_df)
    for date, row in orders_df.iterrows():
        # print(date)
        if date not in prices.index:
            continue

        symbol = row['Symbol']
        shares = row['JPM']
        price = prices.loc[date, row['Symbol']]
        # print(price)

        # iterate the order book here -
        if row['Order'] == 'BUY':
            # --- logic for sell here ---- reference capm#

            trades.loc[date, symbol] += shares * 1
            # Update Cash
            trades.loc[date, 'Cash'] -= (shares * price * (1 + impact))
            trades.loc[date, 'Cash'] -= commission


        if row['Order'] == 'SELL':

            # Update shares
            trades.loc[date, symbol] -= shares * 1
            # Update Cash
            trades.loc[date, 'Cash'] += (shares * price * (1 - impact))
            trades.loc[date, 'Cash'] -= commission



    holdings = trades.copy()

    holdings.loc[start_date, 'Cash'] += start_val
    # print(holdings)

    for i in range(1, len(holdings)):
        holdings.iloc[i] = holdings.iloc[i - 1] + trades.iloc[i]
    # print(holdings)


    portval = holdings.mul(prices, axis='index')

    portval = portval.sum(axis=1)

    return portval


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "tbhalla6"  # replace tb34 with your Georgia Tech username.


def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-01.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals()
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
