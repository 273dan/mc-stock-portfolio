import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt
import numpy as np
import time
import math
import plotly.express as px

# Get start and end dates for stock info
end = dt.datetime.now()
start = end - dt.timedelta(132)



# Initialise session states
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

# Control what stage of the app to currently display
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


# Validate user submitted tickers
def validateStockNames(submission):
    
    # Display correct sections
    setSessionStage(0)
    
    # Lists that will be commited to session states if valid
    search_portfolio = []
    portfolio_data = []
    
    # Reject commas - no stock ticker has commas and it interferes with YF data download
    if ',' in submission:
        
        st.session_state['stocks_valid'] = False
        st.error("Input must not contain ','")
        return False
    
    # Split string of tickers to list of individual tickers
    s_list = submission.split(' ')
    
    for ticker in s_list:
        # Try download data for each ticker
        ticker_data = yf.download(ticker,start,end)['Close']
        
        # If no data -> invalid ticker -> reject
        if ticker_data.empty:
            st.session_state['stocks_valid'] = False
            st.error(f'**{ticker}** not found')
            return False
        else:
            # Reject duplicate tickers
            if ticker in search_portfolio:
                st.error(f'Remove duplicate: {ticker}')
                st.session_state['stocks_valud'] = False
                return False
            
            # For valid tickers, add name and data to list
            search_portfolio.append(ticker)
            portfolio_data.append(ticker_data)
            

    # If all searched tickers made it, commit to session state
    if len(search_portfolio) == len(portfolio_data) == len(s_list):
        
        st.session_state['portfolio'] = search_portfolio
        st.session_state['portfolio_data'] = portfolio_data
        
        # Move on to next stage
        st.session_state['stocks_valid'] = True
        st.success('Valid stock portfolio')

# Validate weight allocation
def validateWeights(weights):
    
    # Display correct sections   
    setSessionStage(1)
    
    # Check if weights sum to 100
    if sum(weights) == 100:
        
        # Commit weights to session state
        st.session_state['portfolio_weights'] = weights
        
        # Move on to next stage
        st.session_state['weights_valid'] = True
        st.success('Valid weight allocation')
        
    else:
        
        # Reject if weights do not sum to 100
        st.session_state['weights_valid'] = False
        st.error(f'Weights must sum to 100. Current sum: {sum(weights)}') 

# Get covariance adjusted random returns for stocks
def getCovReturns(days):
    
    # Get historical daily percentage return in dataframe format
    returns_df = st.session_state['portfolio_data'][0].pct_change()
    for i, row in enumerate(st.session_state['portfolio_data']):
        if i == 0:
            continue
        returns_df = pd.concat((returns_df,row.pct_change()),ignore_index=True,axis=1)
    
    returns_df = returns_df.dropna()
    returns_df.columns = st.session_state['portfolio']
    
    # Calculate covariance
    cov_matrix = returns_df.cov().to_numpy()

    # Apply Cholesky transformation to get lower triangle
    L = np.linalg.cholesky(cov_matrix)

    # Generate random normal variables
    random_norm = np.random.normal(size=(days,len(st.session_state['portfolio'])))

    # Adjust for covariance (by multiplying by lower triangle)
    cor_returns = np.dot(random_norm,L.T)

    # Find average historical return for each stock
    av_returns = returns_df.mean().to_numpy()
    
    # Add average return to covariance adjusted normal variables to give random daily returns
    sim_returns = cor_returns + av_returns
    
    # Return as list of lists
    return sim_returns.T.tolist()
    
    

def runSimulation(sims,days,stake):
    
    # Display correct sections
    setSessionStage(2)
    
    #### For uncorrelated returns ####
    # av_returns = []
    # std_devs = []
    # for close_history in st.session_state['portfolio_data']:
    #     close_history_pct = close_history.pct_change().dropna()
    #     av_returns.append(close_history_pct.mean())
    #     std_devs.append(close_history_pct.std())
    
    # Initialise empty dataframe for simulation results
    sim_df = pd.DataFrame(columns=range(days))
    
    for i in range(sims):
        
        # Initialise empty list for daily portfolio value
        sim_path = []
        
        #### For correlated returns ####
        r_array = getCovReturns(days)
        
        #### For uncorrelated returns ####
        # r_array = []
        # for avr, std in zip(av_returns,std_devs):
        #     returns = np.random.normal(avr,std,days)
        #     r_array.append(returns)
        
        for t in range(days):
            # On day 0 portfolio value is initial value
            if t == 0:
                sim_path.append(stake)
                continue
            
            # Construct todays return by adding each stocks return according to weight
            todays_return = 0
            for i,weight in enumerate(st.session_state['portfolio_weights']):
                todays_return += (weight / 100) * r_array[i][t]
            todays_return += 1
            
            # Value of portfolio on day t is value on day t - 1 * todays return
            sim_path.append(sim_path[t - 1] * todays_return)

        # Prepare and append this simulation's data to simulation results dataframe 
        sim_row = pd.DataFrame([sim_path],columns=sim_df.columns)
        sim_df = pd.concat((sim_df,sim_row),ignore_index=True)
    
    # Add a column to store percentage return on initial investment
    sim_df['Return'] = ((sim_df[days - 1] / stake) - 1) * 100
    
    # Move onto next stage
    st.session_state['sim_results'] = sim_df
    showGraph(sim_df)
    
    
def showGraph(data):
    # Drop return column to avoid graphing it
    data = data.drop(['Return'],axis=1)
    
    # Initialise empty dataframe to add to
    anim_data = pd.DataFrame(columns=data.columns)
    
    # Adjust number of lines to add each frame according to number of simulations
    step = ((data.shape[0] // 50) + 1)
    
    # Use an st.empty to only show newest plot
    with st.empty():
        line = 0
        while line < data.shape[0] + 1:
            
            
            # Enlarge number of simulations plotted with each iteration
            anim_data = data.iloc[:line]
            st.line_chart(anim_data.T,x_label='Day',y_label='Portfolio Value')
            time.sleep(5 / data.shape[0])
            line += step
    # Move on to next stage
    st.session_state['show_results'] = True
    



            




    


st.header('Monte Carlo Stock Portfolio Simulator')

# Form for stock entry - use a form to avoid repeated YF calls
stock_entry = st.form('pick_stock')

with stock_entry:
    stocks_plain = st.text_input('Enter tickers separated by spaces',placeholder='AAPL ^GSPC ^FTSE')
    find_stocks = st.form_submit_button('Find Stocks')

# Validate stocks when form is submitted
if find_stocks:
    validateStockNames(stocks_plain)

# Show next stage
if st.session_state['stocks_valid']:
    
    # Use a form to avoid repeated YF calls
    stock_weight = st.form('pick_weight')
    with stock_weight:
        weights = [st.slider(f'Weight: {stock}',min_value=0,max_value=100,value=100 // len(st.session_state['portfolio'])) for stock in st.session_state['portfolio']]
        pick_weights = st.form_submit_button('Assign Weights')

    # Validate weights when form is submitted
    if pick_weights:
        validateWeights(weights)

# Show next stage
if st.session_state['weights_valid'] and st.session_state['stocks_valid']:
    
    # Use a form so all sim parameters can be adjusted before running sim
    sim_prefs = st.form('sim_prefs')
    with sim_prefs:
        
        # Sliders for number of sims, number of days, and initial portfolio value
        n_sims = st.slider('Number of simulations to run',min_value=5,max_value=500,value=100)
        n_days = st.slider('Number of days ahead to simulate',min_value=100,max_value=1000,value=365)
        pf_val = st.number_input('Initial portfolio value',min_value=0.0,step=1000.0,value=1000.0)
        
        run_sim = st.form_submit_button('Run Simulation')
        
        # Run simulation when button pressed
        if run_sim:
            runSimulation(n_sims,n_days,pf_val)


if st.session_state['show_results']:
    # st.balloons()
    
    # Store last column of dataframe and results column for processing summary stats
    vals = st.session_state['sim_results'][n_days - 1]
    returns = st.session_state['sim_results']['Return']
    
    st.subheader('Summary',divider='gray')
    av, vol, mnmx = st.columns([1/3,1/3,1/3],vertical_alignment='center')
    
    # Get and display average, min, max portfolio values and returns
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
        min_mtr = st.metric(f'Minimum portfolio value',f'£{min_pfval}',f'{min_return}%')
        
    with vol:
        std_pfval = round(vals.std(),2)
        std_return = round(returns.std(),2)
        
        std_mtr = st.metric(f'Volatility (standard deviation)',std_pfval,std_return,delta_color='off')
        
    st.subheader('Distribution')
    hist, qs = st.columns([2/3,1/3],vertical_alignment='center')

    # Plot histogram of portfolio values
    with hist:
        fig = px.histogram(vals, nbins=math.ceil(math.sqrt(n_sims)))
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',paper_bgcolor='rgba(0, 0, 0, 0)',showlegend = False)
        st.plotly_chart(fig)

    with qs:
        # Get and display median, upper quartile and lower quartile portfolio values and returns
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
    
    
    # Compute and display value at risk for 3 different confidence intervals
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

    



    