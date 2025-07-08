import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from util import get_data, plot_data
import datetime as dt
import indicators as indic
from marketsimcode import compute_portvals

"""
Student Name: Tarun Bhalla (replace with your name)  		  	   		 	 	 			  		 			     			  	 
GT User ID: tbhalla6 (replace with your User ID)  		  	   		 	 	 			  		 			     			  	 
GT ID: 904075828 (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""

class ManualStrategy(object):

    def __init__(self, verbose=False, start_val=100000, impact=0.005, commission=9.95):
        self.verbose = verbose
        self.start_val = start_val
        self.impact = impact
        self.commission = commission

    def testPolicy(self,
                   symbol,sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31),
                   startVal=100000):

        prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname='Adj Close')
        prices = prices.fillna(method='ffill').fillna(method='bfill')

        ### ------------ Calculate and get indicators --------------- ###

        #Bollinger bands
        bbp = indic.bollinger_bands(prices, 7).ffill()

        # ema
        ema_short = indic.get_ema(prices, 7).ffill()
        ema_long = indic.get_ema(prices, 21).ffill()
        ema = ema_short > ema_long
        #

        # momentum
        momentum = indic.get_momentum(prices, 10).ffill()

        # rsi
        rsi = indic.get_rsi(prices, 14).ffill()

        #cci
        cci = indic.get_cci(prices, 10)

        # print('BBP', bbp.head(40))
        # print('EMA', ema.head(40))
        # print('RSI', rsi.head(40))
        # print('CCI', cci.head(40))
        # print('momentum', momentum.head(40))
        df_indicators = pd.DataFrame(index=prices.index)
        ### Logic for each signal

        df_indicators["bbp"] = (bbp<0.15).astype(int) - (bbp>0.85).astype(int)
        df_indicators["rsi"] = (rsi<0.3).astype(int) - (rsi>0.7).astype(int)
        df_indicators["momentum"] = (momentum>-0.02).astype(int) - (momentum<0.15).astype(int)
        df_indicators["ema"] = ((ema_short > ema_long).astype(int)
                      - (ema_short < ema_long).astype(int))

        df_indicators["indicators"] = (
            0.4*df_indicators["bbp"]+0.1*df_indicators["rsi"]+0.5*df_indicators["ema"]
        )
        # print(df_indicators.to_string())
        ### SHORT


        trades = pd.DataFrame(data=0.00, index=prices.index, columns=prices.columns)
        curr_holdings = 0

        ### TRADE LOGIC HERE

        for index in range (1, len(df_indicators)):
            ##Buy signal
            if df_indicators['indicators'].iloc[index] > 0.4:
                if curr_holdings == 0:
                    trades.iloc[index] = 1000
                    curr_holdings = 1000
                elif curr_holdings == -1000:
                    trades.iloc[index] = 2000
                    curr_holdings = 1000
            ##Sell signal
            elif df_indicators['indicators'].iloc[index] < -0.4:
                if curr_holdings == 0:
                    trades.iloc[index] = -1000
                    curr_holdings = -1000
                elif curr_holdings == 1000:
                    trades.iloc[index] = -2000
                    curr_holdings = -1000
            else:
                trades.iloc[index] = 0

        # print(trades.to_string())






        # print(trades)
        return trades



    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "tbhalla6"  # replace tb34 with your Georgia Tech username.



if __name__ == '__main__':
    ms = ManualStrategy()
    # In-Sample Test AND OUT SAMPLE ---- data modified here for both
    startval = 100000
    symbol = 'JPM'
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 31)

    benchmarkPrice = get_data([symbol], pd.date_range(start_date, end_date), addSPY=False)
    bm_trades = np.zeros(len(benchmarkPrice))
    bm_trades[0] = 1000
    bm_trades = pd.DataFrame(data=bm_trades, index=benchmarkPrice.index, columns=['Shares'])
    bm_trades = bm_trades.rename(columns={'Shares': 'Trades'})

    bench_vals = compute_portvals(bm_trades)

    bench_dailyret = bench_vals.copy()
    bench_dailyret[1:] = (bench_vals[1:] / bench_dailyret[:-1].values) - 1
    # print(bench_dailyret)
    # print(bench_vals)

    bench_dailyret.iloc[0] = 0
    bench_dailyret = bench_dailyret[1:]
    # print("Daily Return Bench Values:", bench_dailyret)

    cummulative_ret_bench = (bench_vals[-1] - bench_vals[0]) - 1
    # print("Cummulative Returns Bench Values:" ,cummulative_ret_bench)

    bench_mean = bench_dailyret.mean()
    # print("Benchmark Mean:", bench_mean)

    bench_std = bench_dailyret.std()
    # print("Benchmark Standard Deviation:", bench_std)

    manualtrading = ms.testPolicy(symbol, start_date, end_date)
    manualtrading = manualtrading.rename(columns={'JPM': 'Trades'})
    # print(manualtrading.to_string())
    man_vals = compute_portvals(manualtrading)
    # print(man_vals)

    man_dailyret = man_vals.copy()
    man_dailyret[1:] = (man_vals[1:] / man_dailyret[:-1].values) - 1
    # print(bench_dailyret)
    # print(bench_vals)

    man_dailyret.iloc[0] = 0
    man_dailyret = man_dailyret[1:]
    # print("Daily Return Manual Strat Values:", man_dailyret)

    cummulative_ret_manual = (man_vals[-1] - man_vals[0]) - 1
    # print("Cummulative Returns Manual Strat Values:" ,cummulative_ret_bench)

    manual_mean = man_dailyret.mean()
    # print("Manual Strat Mean:", bench_mean)

    manual_std = man_dailyret.std()
    # print("Manual Strat Standard Deviation:", bench_std)

    # normalizing values to return
    norm_benchmark = bench_vals / bench_vals.iloc[0]
    norm_manual = man_vals / man_vals.iloc[0]
    ###TABLE generation
    table_summary = pd.DataFrame({
        'Measure': ['Cummulative Return', 'STD', 'Mean'],
        'Benchmark': [cummulative_ret_bench, bench_std, bench_mean],
        'Manual': [cummulative_ret_manual, manual_std, manual_mean]
    }).set_index('Measure')
    print(table_summary)
    ## Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(norm_benchmark, color='purple', label='Benchmark')
    plt.plot(norm_manual, color='red', label='Manual Strategy')
    plt.xlabel('Date')
    plt.ylabel('Return')
    trade_legend = [
        plt.Line2D([0], [0], color='blue', linestyle='--', label='Long Entry'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Short Entry')
    ]
    # Add  lines for SHORT/LONG
    for date, trade in manualtrading.iterrows():
        if trade["Trades"] == 2000:
            plt.axvline(x=date, color="blue", linestyle="--")
        elif trade["Trades"] == -2000:
            plt.axvline(x=date, color="black", linestyle="--")
    main_legend = plt.legend(loc='upper left', frameon=True)
    plt.gca().add_artist(main_legend)
    plt.legend(handles=trade_legend, loc='upper right', frameon=True)
    plt.title('Manual Trader Vs. Benchmark OUT Sample')
    plt.savefig('ManualVsBenchOutSample.png')
    plt.close()
    ## IN SAMPLE CHART
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    benchmarkPrice = get_data([symbol], pd.date_range(start_date, end_date), addSPY=False)
    bm_trades = np.zeros(len(benchmarkPrice))
    bm_trades[0] = 1000
    bm_trades = pd.DataFrame(data=bm_trades, index=benchmarkPrice.index, columns=['Shares'])
    bm_trades = bm_trades.rename(columns={'Shares': 'Trades'})

    bench_vals = compute_portvals(bm_trades)
    bench_dailyret = bench_vals.copy()
    bench_dailyret[1:] = (bench_vals[1:] / bench_dailyret[:-1].values) - 1
    # print(bench_dailyret)
    # print(bench_vals)

    bench_dailyret.iloc[0] = 0
    bench_dailyret = bench_dailyret[1:]
    # print("Daily Return Bench Values:", bench_dailyret)

    cummulative_ret_bench = (bench_vals[-1] - bench_vals[0]) - 1
    # print("Cummulative Returns Bench Values:" ,cummulative_ret_bench)

    bench_mean = bench_dailyret.mean()
    # print("Benchmark Mean:", bench_mean)

    bench_std = bench_dailyret.std()
    # print("Benchmark Standard Deviation:", bench_std)

    manualtrading = ms.testPolicy(symbol, start_date, end_date)
    manualtrading = manualtrading.rename(columns={'JPM': 'Trades'})
    # print(manualtrading.to_string())
    man_vals = compute_portvals(manualtrading)
    # print(man_vals)

    man_dailyret = man_vals.copy()
    man_dailyret[1:] = (man_vals[1:] / man_dailyret[:-1].values) - 1
    # print(bench_dailyret)
    # print(bench_vals)

    man_dailyret.iloc[0] = 0
    man_dailyret = man_dailyret[1:]
    # print("Daily Return Manual Strat Values:", man_dailyret)

    cummulative_ret_manual = (man_vals[-1] - man_vals[0]) - 1
    # print("Cummulative Returns Manual Strat Values:" ,cummulative_ret_bench)

    manual_mean = man_dailyret.mean()
    # print("Manual Strat Mean:", bench_mean)

    manual_std = man_dailyret.std()
    # print("Manual Strat Standard Deviation:", bench_std)

    # normalizing values to return
    norm_benchmark = bench_vals / bench_vals.iloc[0]
    norm_manual = man_vals / man_vals.iloc[0]
    ###TABLE generation

    table_summary = pd.DataFrame({
        'Measure': ['Cummulative Return', 'STD', 'Mean'],
        'Benchmark': [cummulative_ret_bench, bench_std, bench_mean],
        'Manual': [cummulative_ret_manual, manual_std, manual_mean]
    }).set_index('Measure')
    print(table_summary)

    ## Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(norm_benchmark, color='purple', label='Benchmark')
    plt.plot(norm_manual, color='red', label='Manual Strategy')
    plt.xlabel('Date')
    plt.ylabel('Return')

    trade_legend = [
        plt.Line2D([0], [0], color='blue', linestyle='--', label='Long Entry'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Short Entry')
    ]
    # Add vertical lines for trades
    for date, trade in manualtrading.iterrows():
        if trade["Trades"] == 2000:
            plt.axvline(x=date, color="blue", linestyle="--")
        elif trade["Trades"] == -2000:
            plt.axvline(x=date, color="black", linestyle="--")
    main_legend = plt.legend(loc='upper left', frameon=True)
    plt.gca().add_artist(main_legend)
    plt.legend(handles=trade_legend, loc='upper right', frameon=True)
    plt.title('Manual Trader Vs. Benchmark IN Sample')
    plt.savefig('ManualVsBenchInSample.png')
