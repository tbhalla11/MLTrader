""""""
import numpy as np

"""  		  	   		 	 	 			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
import random  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
import util as ut  		  	   		 	 	 			  		 			     			  	 
import indicators as indic
from marketsimcode import compute_portvals
import BagLearner as bl
import RTLearner as rl
  		  	   		 	 	 			  		 			     			  	 
class StrategyLearner(object):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	 	 			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 	 	 			  		 			     			  	 
    :type verbose: bool  		  	   		 	 	 			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	 	 			  		 			     			  	 
    :type impact: float  		  	   		 	 	 			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	 	 			  		 			     			  	 
    :type commission: float  		  	   		 	 	 			  		 			     			  	 
    """

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "tbhalla6"  # replace tb34 with your Georgia Tech username.

    # constructor  		  	   		 	 	 			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """  		  	   		 	 	 			  		 			     			  	 
        Constructor method  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        self.verbose = verbose  		  	   		 	 	 			  		 			     			  	 
        self.impact = impact  		  	   		 	 	 			  		 			     			  	 
        self.commission = commission
        self.window = 4
        self.leaf_size = 10
        self.bags = 50
        self.learner = bl.BagLearner(
            learner=rl.RTLearner,
            kwargs={"leaf_size": self.leaf_size,},
            bags=self.bags,
            verbose=False
        )
  		  	   		 	 	 			  		 			     			  	 
    # this method should create a QLearner, and train it for trading  		  	   		 	 	 			  		 			     			  	 
    def add_evidence(
        self,
        symbol="IBM",  		  	   		 	 	 			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),
        impact = 0,
        sv=100000):
        """  		  	   		 	 	 			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		 	 	 			  		 			     			  	 
        :type symbol: str  		  	   		 	 	 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
        :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
        :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
        :type sv: int  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        # add your code to do learning here  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        # example usage of the old backward compatible util function  		  	   		 	 	 			  		 			     			  	 
        syms = [symbol]  		  	   		 	 	 			  		 			     			  	 
        dates = pd.date_range(sd, ed)  		  	   		 	 	 			  		 			     			  	 
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		 	 	 			  		 			     			  	 
        prices = prices_all[syms]  # only portfolio symbols  		  	   		 	 	 			  		 			     			  	 
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	 	 			  		 			     			  	 
        # if self.verbose:
        #     print(prices)


        # example use with new colname  		  	   		 	 	 			  		 			     			  	 
        volume_all = ut.get_data(  		  	   		 	 	 			  		 			     			  	 
            syms, dates, colname="Volume"  		  	   		 	 	 			  		 			     			  	 
        )  # automatically adds SPY  		  	   		 	 	 			  		 			     			  	 
        volume = volume_all[syms]  # only portfolio symbols  		  	   		 	 	 			  		 			     			  	 
        volume_SPY = volume_all["SPY"]  # only SPY, for comparison later  		  	   		 	 	 			  		 			     			  	 
        # if self.verbose:
        #     print(volume)
        prices = prices.bfill().ffill()

        ### Indicators:
        bbp = indic.bollinger_bands(prices, 7).ffill()
        # ema
        ema_short = indic.get_ema(prices, 7).ffill()
        ema_long = indic.get_ema(prices, 21).ffill()
        ema = (ema_short - ema_long) / ema_long


        # momentum
        momentum = indic.get_momentum(prices, 10).ffill()

        # rsi
        rsi = indic.get_rsi(prices, 14).ffill()

        # cci
        cci = indic.get_cci(prices, 10)

        df_indicators = pd.DataFrame(index=prices.index)
        ### Logic for each signal

        df_indicators["bbp"] = (bbp < 0.15).astype(int) - (bbp > 0.85).astype(int)
        df_indicators["rsi"] = (rsi < 0.3).astype(int) - (rsi > 0.7).astype(int)
        df_indicators["momentum"] = (momentum > -0.02).astype(int) - (momentum < 0.15).astype(int)
        df_indicators["ema"] = ((ema_short > ema_long).astype(int)
                                - (ema_short < ema_long).astype(int))

        x_train = (
                0.45 * df_indicators["bbp"] + 0.25 * df_indicators["rsi"] + 0.35 * df_indicators["ema"]
        ).to_frame()
        x_train.columns=["Indicator"]
        x_train = x_train[:-5]  # Lookahead adjustment
        self.train_mean = x_train.mean()
        self.train_std = x_train.std()
        x_train = (x_train - self.train_mean) / (self.train_std + 1e-6)
        y_train = np.zeros(x_train.shape[0])
        future_returns = prices[symbol].pct_change(5).shift(-5)
        # 5-day forward returns

        buy_threshold = 0.5 + self.impact
        sell_threshold = -0.04 - self.impact

        for index in range(len(x_train)):
            fut_ret = future_returns.iloc[index]

            if fut_ret > buy_threshold:
                y_train[index] = 1
                ## BUY
            elif fut_ret < sell_threshold:
                y_train[index] = -1
                ## SELL
            else:
                y_train[index] = 0

        # print("y_train distribution:", np.unique(y_train, return_counts=True))

        self.learner.add_evidence(x_train.values, y_train)



    # this method should use the existing policy and test it against new data
    def testPolicy(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        symbol="IBM",  		  	   		 	 	 			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2010, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        sv=10000,  		  	   		 	 	 			  		 			     			  	 
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 	 	 			  		 			     			  	 
        :type symbol: str  		  	   		 	 	 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
        :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
        :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
        :type sv: int  		  	   		 	 	 			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	 	 			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	 	 			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	 	 			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	 	 			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        # here we build a fake set of trades  		  	   		 	 	 			  		 			     			  	 
        # your code should return the same sort of data  		  	   		 	 	 			  		 			     			  	 
        dates = pd.date_range(sd, ed)  		  	   		 	 	 			  		 			     			  	 
        prices = ut.get_data([symbol], dates, addSPY=False)  # automatically adds SPY
        trades = prices[[symbol]].copy()
        trades.loc[:] = 0
        # print(trades)

        prices = prices.bfill().ffill()

        ### Indicators:
        bbp = indic.bollinger_bands(prices, 7).ffill()
        # ema
        ema_short = indic.get_ema(prices, 7).ffill()
        ema_long = indic.get_ema(prices, 21).ffill()
        ema = (ema_short - ema_long) / ema_long

        # momentum
        momentum = indic.get_momentum(prices, 10).ffill()

        # rsi
        rsi = indic.get_rsi(prices, 14).ffill()


        # cci
        cci = indic.get_cci(prices, 10)
        # print("BBP range:", bbp.min(), bbp.max())
        # print("RSI range:", rsi.min(), rsi.max())
        # print("EMA ratio range:", ema.min(), ema.max())
        df_indicators = pd.DataFrame(index=prices.index)

        df_indicators["bbp"] = (bbp < 0.15).astype(int) - (bbp > 0.85).astype(int)
        df_indicators["rsi"] = (rsi < 0.3).astype(int) - (rsi > 0.7).astype(int)
        df_indicators["momentum"] = (momentum > -0.02).astype(int) - (momentum < 0.15).astype(int)
        df_indicators["ema"] = ((ema_short > ema_long).astype(int)
                                - (ema_short < ema_long).astype(int))
        x_test = (
                0.5 * df_indicators["bbp"] + 0.2 * df_indicators["rsi"] + 0.3 * df_indicators["ema"]
        ).to_frame()
        x_test.columns = ["Indicator"]
        # print(x_test)


        y_pred = (self.learner.query(x_test.values))

        # print(y_pred)

        curr_holdings = 0
        for index in range(1, len(prices)):
            ##Buy signal
            if y_pred[index] > 0.05:
                if curr_holdings == 0:
                    trades.iloc[index] = 1000
                    curr_holdings = 1000
                elif curr_holdings == -1000:
                    trades.iloc[index] = 2000
                    curr_holdings = 1000
            ##Sell signal
            elif y_pred[index] < 0:
                if curr_holdings == 0:
                    trades.iloc[index] = -1000
                    curr_holdings = -1000
                elif curr_holdings == 1000:
                    trades.iloc[index] = -2000
                    curr_holdings = -1000
            else:
                trades.iloc[index] = 0

        return trades

if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    print("One does not simply think up a strategy")




