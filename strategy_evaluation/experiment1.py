import datetime as dt
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from util import get_data, plot_data
import indicators as indic
from marketsimcode import compute_portvals
import StrategyLearner as sl
import ManualStrategy as ml



def author():
    return 'tbhalla6'

def experiment1():

    ##### BENCHMARK DATA #####  IN SAMPLE ##########
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    date = pd.date_range(start_date, end_date)
    symbol = "JPM"
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    df_benchmark = pd.DataFrame(index=prices.index, columns=['Trades'])
    df_benchmark['Trades'] = 0

    df_benchmark.loc[df_benchmark.index.min(), 'Trades'] = 1000
    bench_port = compute_portvals(df_benchmark, 100000, 9.95, 0.005)
    #nomralize portfolio
    bench_port = bench_port/bench_port.iloc[0]

    ####### STRATEGY LEARNER ######

    learner = sl.StrategyLearner()
    learner.add_evidence(symbol, start_date, end_date)
    df_strat = learner.testPolicy("JPM", start_date, end_date)
    # print(df_strat.head(50))
    df_strat = df_strat.rename(columns={'JPM': 'Trades'})
    strat_port = compute_portvals(df_strat, 100000, 9.95, 0.005)
    strat_port = strat_port/strat_port.iloc[0]
    # print(strat_port)


    ######## MANUAL STRATEGY ######

    mlearner = ml.ManualStrategy()
    df_manual= mlearner.testPolicy(symbol, start_date, end_date)
    df_manual = df_manual.rename(columns={'JPM': 'Trades'})
    manual_port = compute_portvals(df_manual, 100000, 9.95, 0.005)
    manual_port = manual_port/manual_port.iloc[0]

    ###### PLOTTNG THE ABOVE ####

    plt.figure(figsize=(14, 7))
    plt.plot( bench_port, label='Benchmark - HOLD', color='blue')
    plt.plot( strat_port, label='Strategy Learner - Classification', color='red')
    plt.plot( manual_port, label='Manual Strategy', color='green')
    plt.title("Comparing Performance of Strategy Learner vs Manual Strategy vs Benchmark")
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Portfolio value')
    plt.savefig('Experiment1')
    plt.close()



    ##OUT OF SAMPLE ###############
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 31)
    date = pd.date_range(start_date, end_date)
    symbol = "JPM"
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    df_benchmark = pd.DataFrame(index=prices.index, columns=['Trades'])
    df_benchmark['Trades'] = 0

    df_benchmark.loc[df_benchmark.index.min(), 'Trades'] = 1000
    bench_port = compute_portvals(df_benchmark, 100000, 9.95, 0.005)
    # nomralize portfolio
    bench_port = bench_port / bench_port.iloc[0]

    ####### STRATEGY LEARNER ######

    learner = sl.StrategyLearner()
    learner.add_evidence(symbol, start_date, end_date)
    df_strat = learner.testPolicy("JPM", start_date, end_date)
    # print(df_strat.head(50))
    df_strat = df_strat.rename(columns={'JPM': 'Trades'})
    strat_port = compute_portvals(df_strat, 100000, 9.95, 0.005)
    strat_port = strat_port / strat_port.iloc[0]
    # print(strat_port)

    ######## MANUAL STRATEGY ######

    mlearner = ml.ManualStrategy()
    df_manual = mlearner.testPolicy(symbol, start_date, end_date)
    df_manual = df_manual.rename(columns={'JPM': 'Trades'})
    manual_port = compute_portvals(df_manual, 100000, 9.95, 0.005)
    manual_port = manual_port / manual_port.iloc[0]

    ###### PLOTTNG THE ABOVE ####

    plt.figure(figsize=(14, 7))
    plt.plot(bench_port, label='Benchmark - HOLD', color='blue')
    plt.plot(strat_port, label='Strategy Learner - Classification', color='red')
    plt.plot(manual_port, label='Manual Strategy', color='green')
    plt.title("Comparing Performance of Strategy Learner vs Manual Strategy vs Benchmark OUT OF SAMPLE")
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Portfolio value')
    # plt.savefig('Experiment1_outsample')

    ### CHARTS #####
    c_ret_benchmark = (bench_port.iloc[-1] / bench_port.iloc[0]) - 1
    c_ret_manual = (manual_port.iloc[-1] / manual_port.iloc[0]) - 1
    c_ret_strat = (strat_port.iloc[-1] / strat_port.iloc[0]) - 1

    b_daily = bench_port.pct_change().mean()
    m_daily = manual_port.pct_change().mean()
    s_daily = strat_port.pct_change().mean()

    b_std = bench_port.pct_change().std()
    m_std = manual_port.pct_change().std()
    s_std = strat_port.pct_change().std()


    table_summary = pd.DataFrame({
        'Measure': ['Cumulative Return', 'Mean Daily Return', 'Volatility (STD)'],
        'Benchmark': [c_ret_benchmark, b_daily, b_std],
        'Manual': [c_ret_manual, m_daily, m_std],
        'Strategy': [c_ret_strat, s_daily, s_std]
    }).set_index('Measure')

    print(table_summary)




if __name__ == '__main__':

    experiment1()