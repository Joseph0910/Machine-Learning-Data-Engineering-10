class PortfolioOptimization():
	def __init__(self,matrix,maptype,data,Covariance,Correlation,
				pair, portfolio_weights, meanR, cov,combo,portf_type,
				Rf,returns,n_opt_trials,n_jobs,trial):
		self.matrix = matrix
		self.maptype = maptype
		self.data = data
		self.Covariance = Covariance
		self.Correlation = Correlation
		self.pair = pair
		self.portfolio_weights = portfolio_weights
		self.meanR = meanR		
		self.cov = cov
		self.combo = combo
		self.portf_type = portf_type
		self.Rf = Rf
		self.returns = returns
		self.n_opt_trials = n_opt_trials
		self.n_jobs = n_jobs
		self.trial = trial

	def portfolio_eval():

		# call linear correlation module of stocks #
		# uncorrelated stocks should be grouped together to minimize potential losses #
		# also look at covariance of a portfolio < average covariance # 
		# concept here is minimal risk, but optimal quarterly dividends #

		# threshold to be added to identify 'Uncorrelated' and 'Correlated' stocks #

		Covariance = returns.cov()
		Correlation = returns.corr()
		np.round(Correlation,3)

	def plot1():
	    fig = plt.figure(figsize=(10,10))
	    ax1 = fig.add_subplot(111)
	    cmap = cm.get_cmap('jet', 30)
	    if (maptype == 'cmap'):
	        cax = ax1.imshow(matrix, interpolation="nearest", cmap=cmap, alpha=0.7)
	    else:
	        cax = ax1.imshow(matrix, interpolation="nearest", cmap=maptype, alpha=0.7)
	    ax1.grid(True)
	    plt.title('Stocks Correlation')
	    ax1.set_xticks(np.arange(len(companies)))
	    ax1.set_yticks(np.arange(len(companies)))
	    ax1.set_xticklabels(companies,fontsize=10,rotation=90)
	    ax1.set_yticklabels(companies,fontsize=10)
	    ax1.set_alpha(0.4)
	    ticks = numpy.arange(0,9,1)
	    fig.colorbar(cax)
	    plt.show()

	def plot2():

		mean_Covariance = Covariance.mean()

	    Pairable = np.zeros(Covariance.shape)
	    plt.figure(figsize=(16,7))

	    for i in range(len(companies)):
	        for j in range(len(companies)-i):
	            if(Covariance[i,j] > mean_Covariance[i] or Covariance[i,j] > mean_Covariance[j] or Correlation[i][j]>0.5):
	                plt.plot(i, j, 'o', color='green', alpha=0.5) 
	            else:
	                plt.plot(i, j, 'o', color='blue', alpha=0.5)
	                Pairable[i,j] = 1

	    plt.xlim(-1,len(companies)+1)
	    plt.ylim(-1,len(companies)+1)
	    plt.xticks(range(len(companies)), companies, rotation=40)    
	    plt.yticks(range(len(companies)), companies)
	    #plt.set_xticklabels(companies,fontsize=10,rotation=40)
	    plt.legend()

	    return Pairable



	def portfolio_metrics():
		# sharpe ratio (risk adj returns) : >1 is good, >2 is great, >3 is excellent. risk free rate should be tbill rate
		# sortino ratio (downside volatility only, as opposed to general volatility which includes upside and downside) 
		# calmar ratio

		# calculate sharpe ratio for each portfolio of uncorrelated stock combinations #
		# based on sharpe ratio, divide portfolios into good, better, and best 

		# column 1: portfolio stocks #
		# column 2: weights # 
		# column 3: sharpe #
		# column 4 : portfolio return # 
		# column 5: portfolio volatility #

		stocks_rng = range(len(companies))

		Rp = portfolio_weights.dot(meanR.T) 
		SigmaP = portfolio_weights.dot(cov.dot(portfolio_weights.T)) * len(returns)
		s_ratio = (Rp - Rf)/np.sqrt(SigmaP)

		sortino = sortino_ratio(returns)
		calmar = calmar_ratio(returns)
		omega = omega_ratio(returns)

		return s_ratio,sortino,calmar,omega


		# portfolio_volatility = np.sqrt(np.dot(portfolio_weights.T, 
		#                             np.dot(cov_mat_annual, portfolio_weights)))

	    not_in_pair = []
	    for i in stocks_rng:
	        if(i not in pair):
	            not_in_pair.append(i)
	        else:
	            continue
	            
	    for tick in not_in_pair:
	        total_pair = len(pair)
	        for i in pair:
	            if(Pairable[tick,i]!=1 or Pairable[i,tick]!=1):
	                total_pair -= 1
	        if(float(total_pair)/len(pair) > 0.5):
	            pair.add(tick)              
	    
	    return pair


	def portfolio_math():
		# the portfolio with the highest spread between return and sharpe ratio #

		portfolio_collection = []
		optim = {'good':[],'better':[],'best': []}


	    pair = set(combo)
	    pair = check_pairs(pair)
	    #print("new pair by checking pairable: ", pair)
	    if pair in portfolio_collection:
	        #print('returning')
	        return
	    portfolio_collection.append(pair)
	    #print("portfoilio: ", portfolio_collection)
	    sharpe_r = 0
	    eff_weights = np.ones(len(pair))

	    symbols = [companies[s] for s in pair]        
	    mean_returns = np.array(returns[symbols].mean()) * len(returns)
	    sub_cov_mat = np.array(returns[symbols].cov())        

	    for _ in range(200):
	        weights = [np.random.randint(50,500) for _ in pair]
	        weights = np.array(weights, dtype=float)
	        weights /= weights.sum()
	        s_r = sharpe_ratio(pair, weights, mean_returns, sub_cov_mat)
	        if( s_r > sharpe_r):
	            sharpe_r = s_r
	            eff_weights = weights
	        
	    if (sharpe_r >= 1 and sharpe_r < 2) :
	        optim['good'].append([[companies[s] for s in pair],eff_weights,sharpe_r])
	    if (sharpe_r >= 2 and sharpe_r < 3) :
	        optim['better'].append([[companies[s] for s in pair],eff_weights,sharpe_r])
	    if(sharpe_r >=3) :
	        optim['best'].append([[companies[s] for s in pair],eff_weights,sharpe_r])

		count=0
		run = list(combinations(stocks_rng,2))
		for combo in run:
		    count +=1
		    if(Pairable[combo[0],combo[1]]!=1 or Pairable[combo[1],combo[0]]!=1):
		        continue
		    else:
		        #print("taking the combo")
		        select(combo)
		print("number of combinations: ", count)

	def portfolio_summ(self):

		portf_type.columns = ['Portfolio', 'Weights', 'Sharpe Ratio']

		portfolio_return = []

		for row in portf_type.iterrows():
		    #print(row[1][0])
		    mean = np.array(returns[row[1][0]].mean()) * len(returns)
		    #mean = ((1+np.mean(returns[row[1][0]]))**252)-1 #annulazied return for 252 trading days
		    portfolio_weights = np.array(row[1][1])
		    Rp = portfolio_weights.dot(mean.T) 
		    portfolio_return.append(round(Rp * 100, 2))
		    #print('Rp:', Rp)
		    
		portf_type['Portfolio Return'] = portfolio_return

		portfolio_volatility = []

		for row in portf_type.iterrows():
		    portfolio = row[1][0]
		    portfolio_weights = row[1][1]
		    portfolio_data = data[portfolio]
		    portfolio = portfolio_data.mul(portfolio_weights,axis=1).sum(axis=1)
		    volatility = np.std(portfolio)
		    portfolio_volatility.append(volatility)

		portf_type['Portfolio Volatility'] = portfolio_volatility
		return portf_type # should be a table with portfolio combinations (rows) and stats (column)

	def optimize_fn(self):

		def optimize_ppo2(trial):
		    return {
		        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
		        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
		        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
		        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
		        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
		        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
		        'lam': trial.suggest_uniform('lam', 0.8, 1.)
		    }

	    def optimize_envs(trial):
	    return {
	        'reward_len': int(trial.suggest_loguniform('reward_len', 1, 200)),
	        'forecast_len': int(trial.suggest_loguniform('forecast_len', 1, 200)),
	        'confidence_interval': trial.suggest_uniform('confidence_interval', 0.7, 0.99),
	    }


	    env_params = optimize_envs(trial)
	    agent_params = optimize_ppo2(trial)
	    
	    train_env, validation_env = initialize_envs(**env_params) 
	    model = PPO2(MlpLstmPolicy, train_env, **agent_params) # cost we return from our function is the average reward over the testing period # 
	    
	    model.learn(len(train_env.df))
	    
	    rewards, done = [], False

	    obs = validation_env.reset()
	    for i in range(len(validation_env.df)):
	        action, _ = model.predict(obs)
	        obs, reward, done, _ = validation_env.step(action)
	        rewards += reward
	    
	    return -np.mean(rewards)

	def opt_execution(self):
	    study = optuna.create_study(study_name='optimize_profit', storage='sqlite:///params.db', load_if_exists=True)
	    study.optimize(objective_fn, n_trials=n_opt_trials, n_jobs=n_jobs)


	def opt_results(self):
		study = optuna.load_study(study_name='optimize_profit', storage='sqlite:///params.db')
		params = study.best_trial.params

		env_params = {
		    'reward_len': int(params['reward_len']),
		    'forecast_len': int(params['forecast_len']),
		    'confidence_interval': params['confidence_interval']
		}

		train_env = DummyVecEnv([lambda: BitcoinTradingEnv(train_df, **env_params)])

		model_params = {
		    'n_steps': int(params['n_steps']),
		    'gamma': params['gamma'],
		    'learning_rate': params['learning_rate'],
		    'ent_coef': params['ent_coef'],
		    'cliprange': params['cliprange'],
		    'noptepochs': int(params['noptepochs']),
		    'lam': params['lam']
		}

		model = PPO2(MlpLstmPolicy, train_env, **model_params)


	def plot_portf():
		portf_type['PortfolioL'] = portf_type['Portfolio'].apply(lambda x: ','.join(map(str, x)))
		plt.figure(figsize=(14,5))
		fig, ax1 = plt.subplots(figsize=(14,6))

		ax2 = ax1.twinx()
		ax2.plot(portf_type['PortfolioL'], portf_type['Sharpe Ratio'], 'g-', alpha=0.6, marker='o', 
		         linestyle='dashed', linewidth=2, markersize=6)
		ax1.plot(portf_type['PortfolioL'], portf_type['Portfolio Return'], 'b-', alpha=0.6, marker='o', 
		         linestyle='dashed', linewidth=2, markersize=6)
		ax1.plot(portf_type['PortfolioL'],portf_type['Portfolio Volatility'], 'r-', alpha=0.6, marker='o', 
		         linestyle='dashed', linewidth=2, markersize=6)

		ax1.set_ylabel('Return and Volatility')
		ax2.set_ylabel('Sharpe Ratio')
		ax1.xaxis.set_tick_params(rotation=90)
		ax1.legend(loc='upper left')
		ax2.legend(loc='upper right')

		#ax2.set_alpha(0.4)
		#ax1.set_alpha(0.4)
		plt.show()

# execute #
plot1(Correlation,'cmap')

Pairable = plot2(data,np.array(Covariance), np.array(Correlation))

optim['good']
optim['better']
optim['best']
portfolio_collection

