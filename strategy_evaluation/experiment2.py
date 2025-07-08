import datetime as dt
from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import marketsimcode as msc
import StrategyLearner
from util import get_data
from marketsimcode import compute_portvals

def author():
    return 'tbhalla6'


def calculate_metrics(portvals):
    daily_rets = portvals.pct_change().dropna()
    cum_ret = (portvals[-1] / portvals[0]) - 1
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sharpe_ratio = np.sqrt(252) * avg_daily_ret / std_daily_ret

    return {
        'Cumulative Return': cum_ret,
        'Avg Daily Return': avg_daily_ret,
        'Volatility': std_daily_ret,
        'Sharpe Ratio': sharpe_ratio

    }

def experiment2():
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Auto-detect terminal width
    pd.set_option('display.max_colwidth', None)  # Show full column content
    pd.set_option('display.float_format', '{:.6f}'.format)  # Show 6 decimal places

    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 30)
    symbol = "JPM"

    learner = StrategyLearner.StrategyLearner(verbose=True, impact=0.005, commission=0)
    learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date, sv=100000)
    trades = learner.testPolicy(symbol=symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)

    trades.columns = ['Trades']

    portvals_min_impact = compute_portvals(trades, 100000, 0, 0.0)

    portvals_low_med_impact = compute_portvals(trades, 100000, 0, 0.0025)

    portvals_regular_impact = compute_portvals(trades, 100000, 0, 0.005)

    portvals_mid_high_impact = compute_portvals(trades, 100000, 0, 0.05)

    portvals_high_impact = compute_portvals(trades, 100000, 0, 0.1)

    daily_returns = {
        'Min Impact': portvals_min_impact[1:] / portvals_min_impact[:-1].values - 1,
        'Low-Med Impact': portvals_low_med_impact[1:] / portvals_low_med_impact[:-1].values - 1,
        'Regular Impact': portvals_regular_impact[1:] / portvals_regular_impact[:-1].values - 1,
        'Mid-High Impact': portvals_mid_high_impact[1:] / portvals_mid_high_impact[:-1].values - 1,
        'High Impact': portvals_high_impact[1:] / portvals_high_impact[:-1].values - 1
    }

    daily_returns = pd.DataFrame(daily_returns)
    daily_mean = daily_returns.agg(['mean']).T

    plt.figure(figsize=(12, 6))
    bars = plt.barh(daily_mean.index, daily_mean['mean'], 0.6,
                    color=['b', 'r', 'g', 'y', 'purple'])

    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('Daily Returns based on Impact %')
    plt.xlabel('Average Daily Return')
    plt.grid(axis='x', alpha=0.3)
    colors = ['b', 'r', 'g', 'y', 'purple']
    labels = ['Min Impact - 0.0', 'Low-Med Impact - 0.0025', 'Regular Impact - 0.005', 'Mid-High Impact - 0.05', 'High Impact - 0.1']
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color, label=label)
                       for color, label in zip(colors, labels)]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.savefig('Experiment2 - Impact in relation to Strategy Learner')

    plt.close()

    cumulative_returns = (1 + daily_returns).cumprod()


    plt.figure(figsize=(14, 7))


    colors = ['b', 'g', 'r', 'black', 'y']

    linewidths = [2.5, 2, 1.5, 1, 2]

    for i, col in enumerate(cumulative_returns.columns):
        plt.plot(cumulative_returns.index,
                 cumulative_returns[col],
                 color=colors[i],
                 linewidth=linewidths[i],
                 label=col)

    # Formatting
    plt.title('Cumulative Returns of 1$ Based on Impact')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Returns')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.savefig('Experiment2 - Cumulative Returns of Strategy Learner based on 1$ investment')

    plt.close()

    ### CHARTS ###
    impacts = {
        'Min Impact (0.0%)': 0.0,
        'Low Impact (0.25%)': 0.0025,
        'Regular Impact (0.5%)': 0.005,
        'High Impact (5%)': 0.05,
        'Very High Impact (10%)': 0.1
    }
    portvals = {name: compute_portvals(trades, 100000, 0, impact)
                for name, impact in impacts.items()}
    metrics = pd.DataFrame({name: calculate_metrics(pvals)
                            for name, pvals in portvals.items()}).T
    print(metrics)

if __name__ == '__main__':

    experiment2()