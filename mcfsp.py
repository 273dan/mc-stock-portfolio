import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt
import numpy as np
import time
import math
import plotly.express as px

end = dt.datetime.now()
start = end - dt.timedelta(100)




if 'stocks_valid' not in st.session_state:
    st.session_state['stocks_valid'] = False
if 'weights_valid' not in st.session_state:
    st.session_state['weights_valid'] = False
if 'show_results' not in st.session_state:
    st.session_state['show_results'] = False
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []
if 'portfolio_data' not in st.session_state:
    st.session_state['portfolio_data'] = []
if 'portfolio_weights' not in st.session_state:
    st.session_state['portfolio_weights'] = []
if 'sim_results' not in st.session_state:
    st.session_state['sim_results'] = pd.DataFrame()

def setSessionStage(stage):
    if stage == 0:
        st.session_state['stocks_valid'] = False
        st.session_state['weights_valid'] = False
        st.session_state['show_results'] = False
    if stage == 1:
        st.session_state['weights_valid'] = False
        st.session_state['show_results'] = False
    if stage == 2:
        st.session_state['show_results'] = False



def validateStockNames(submission):
    
    setSessionStage(0)
    search_portfolio = []
    portfolio_data = []
    if ',' in submission:
        st.session_state['stocks_valid'] = False
        st.error("Input must not contain ','")
        return False
    s_list = submission.split(' ')
    for ticker in s_list:
        ticker_data = yf.download(ticker,start,end)['Close']
        if ticker_data.empty:
            st.session_state['stocks_valid'] = False
            st.error(f'**{ticker}** not found')
            return False
        else:
            if ticker in search_portfolio:
                st.error(f'Remove duplicate: {ticker}')
                st.session_state['stocks_valud'] = False
                return False
            search_portfolio.append(ticker)
            portfolio_data.append(ticker_data)
            

    if len(search_portfolio) == len(portfolio_data) == len(s_list):
        
        st.session_state['portfolio'] = search_portfolio
        st.session_state['portfolio_data'] = portfolio_data
        
        st.session_state['stocks_valid'] = True
        st.success('Valid stock portfolio')

def validateWeights(weights):
    setSessionStage(1)
    if sum(weights) == 100:
        st.session_state['portfolio_weights'] = weights
        st.session_state['weights_valid'] = True
        st.success('Valid weight assignment')
        
    else:
        st.session_state['weights_valid'] = False
        st.error(f'Weights must sum to 100. Current sum: {sum(weights)}') 


def getCovReturns(days):
    returns_df = st.session_state['portfolio_data'][0].pct_change()
    for i, row in enumerate(st.session_state['portfolio_data']):
        if i == 0:
            continue
        returns_df = pd.concat((returns_df,row.pct_change()),ignore_index=True,axis=1)
    
    returns_df = returns_df.dropna()
    returns_df.columns = st.session_state['portfolio']
    
    cov_matrix = returns_df.cov().to_numpy()

    L = np.linalg.cholesky(cov_matrix)

    random_norm = np.random.normal(size=(days,len(st.session_state['portfolio'])))

    cor_returns = np.dot(random_norm,L.T)

    av_returns = returns_df.mean().to_numpy()

    sim_returns = cor_returns + av_returns

    return sim_returns.T.tolist()
    
    

def runSimulation(sims,days,stake):
    setSessionStage(2)
    
    
    
    
    av_returns = []
    std_devs = []
    for close_history in st.session_state['portfolio_data']:
        close_history_pct = close_history.pct_change().dropna()
        av_returns.append(close_history_pct.mean())
        std_devs.append(close_history_pct.std())
    
    sim_df = pd.DataFrame(columns=range(days))
    
    for i in range(sims):
        sim_path = []
        
        # For correlated returns
        r_array = getCovReturns(days)
        
        # For uncorrelated returns:
        # r_array = []
        # for avr, std in zip(av_returns,std_devs):
        #     returns = np.random.normal(avr,std,days)
        #     r_array.append(returns)
        
        for t in range(days):
            if t == 0:
                sim_path.append(stake)
                continue
            todays_return = 0
            for i,weight in enumerate(st.session_state['portfolio_weights']):
                todays_return += (weight / 100) * r_array[i][t]
            todays_return += 1
            sim_path.append(sim_path[t - 1] * todays_return)

        sim_row = pd.DataFrame([sim_path],columns=sim_df.columns)
        sim_df = pd.concat((sim_df,sim_row),ignore_index=True)
    
    sim_df['Return'] = ((sim_df[days - 1] / stake) - 1) * 100
    # st.write(sim_df)
    st.session_state['sim_results'] = sim_df
    
    showGraph(sim_df)
    
    
def showGraph(data):
    data = data.drop(['Return'],axis=1)
    anim_data = pd.DataFrame(columns=data.columns)
    step = ((data.shape[0] // 50) + 1)
    
    with st.empty():
        line = 0
        while line < data.shape[0] + 1:
            
            anim_data = data.iloc[:line]
            #st.write(anim_data)
            st.line_chart(anim_data.T,x_label='Day',y_label='Portfolio Value')
            time.sleep(5 / data.shape[0])
            line += step
    st.session_state['show_results'] = True
    



            




    


st.header('Monte Carlo Stock Portfolio Simulator')

stock_entry = st.form('pick_stock')

with stock_entry:
    stocks_plain = st.text_input('Enter tickers separated by spaces',placeholder='AAPL ^GSPC ^FTSE')
    find_stocks = st.form_submit_button('Find Stocks')

if find_stocks:
    validateStockNames(stocks_plain)

if st.session_state['stocks_valid']:
    stock_weight = st.form('pick_weight')
    with stock_weight:
        weights = [st.slider(f'Weight: {stock}',min_value=0,max_value=100,value=100 // len(st.session_state['portfolio'])) for stock in st.session_state['portfolio']]
        pick_weights = st.form_submit_button('Assign Weights')

    if pick_weights:
        validateWeights(weights)

if st.session_state['weights_valid'] and st.session_state['stocks_valid']:
    sim_prefs = st.form('sim_prefs')
    with sim_prefs:
        n_sims = st.slider('Number of simulations to run',min_value=5,max_value=500,value=100)
        n_days = st.slider('Number of days ahead to simulate',min_value=100,max_value=1000,value=365)
        pf_val = st.number_input('Initial portfolio value',min_value=0.0,step=1000.0,value=1000.0)
        run_sim = st.form_submit_button('Run Simulation')
        if run_sim:
            runSimulation(n_sims,n_days,pf_val)


if st.session_state['show_results']:
    # st.balloons()
    vals = st.session_state['sim_results'][n_days - 1]
    returns = st.session_state['sim_results']['Return']
    
    st.subheader('Summary',divider='gray')
    av, vol, mnmx = st.columns([1/3,1/3,1/3],vertical_alignment='center')
    
    average_pfval = round(vals.mean(),2)
    av_return = round(returns.mean(),2)

    max_pfval = round(vals.max(),2)
    max_return = round(returns.max() ,2)

    min_pfval = round(vals.min(),2)
    min_return = round(returns.min(),2)
    
    with av:
        av_mtr = st.metric(f'Mean portfolio value after day {n_days}',f'£{average_pfval}',f'{av_return}%')
    with mnmx:
        max_mtr = st.metric(f'Maximum portfolio value',f'£{max_pfval}',f'{max_return}%')
        min_mtr = st.metric(f'Maximum portfolio value',f'£{min_pfval}',f'{min_return}%')
        
    with vol:
        std_pfval = round(vals.std(),2)
        std_return = round(returns.std(),2)
        
        std_mtr = st.metric(f'Volatility (standard deviation)',std_pfval,std_return,delta_color='off')
        
    st.subheader('Distribution')
    hist, qs = st.columns([2/3,1/3],vertical_alignment='center')

    with hist:
        fig = px.histogram(vals, nbins=math.ceil(math.sqrt(n_sims)))
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',paper_bgcolor='rgba(0, 0, 0, 0)',showlegend = False)
        st.plotly_chart(fig)

    with qs:
        med_val = round(vals.median(),2)
        med_return = round(returns.median(),2)
        
        uq_val = round(vals.quantile(0.75),2)
        uq_return = round(returns.quantile(0.75),2)

        lq_val = round(vals.quantile(0.25),2)
        lq_return = round(returns.quantile(0.25),2)

        med_mtr = st.metric(f'Median portfolio value',f'£{med_val}',f'{med_return}%')
        uq_mtr = st.metric(f'Upper quartile portfolio value',f'£{uq_val}',f'{uq_return}$')
        lq_mtr = st.metric(f'Lower quartile portfolio value',f'£{lq_val}',f'{lq_return}%')
    

    st.subheader('Risk')
    v90, v95, v99 = st.columns([1/3,1/3,1/3])
    
    var90 = round(np.percentile(vals,10),2)
    var_return90 = round(np.percentile(returns,10),2)
    
    var95 = round(np.percentile(vals,5),2)
    var_return95 = round(np.percentile(returns,5),2)
    
    var99 = round(np.percentile(vals,1),2)
    var_return99 = round(np.percentile(returns,1),2)
    
    with v90:
        st.metric('90% Value at risk',f'£{var90}',f'{var_return90}%')
    with v95:
        st.metric('95% Value at risk',f'£{var95}',f'{var_return95}%')
    with v99:
        st.metric('99% Value at risk',f'£{var99}',f'{var_return99}%')

    



    