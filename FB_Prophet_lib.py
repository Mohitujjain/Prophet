#!/usr/bin/env python
# coding: utf-8

# # Fbprophet 
# 
# Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.
# 
# Prophet is open source software released by Facebook’s Core Data Science team. It is available for download on CRAN and PyPI.
# 
# ###  Accurate and Fast
# 
# Prophet is used in many applications across Facebook for producing reliable forecasts for planning and goal setting. We’ve found it to perform better than any other approach in the majority of cases. We fit models in Stan so that you get forecasts in just a few seconds.
# 
# ### Fully Automatic
# 
# Get a reasonable forecast on messy data with no manual effort. Prophet is robust to outliers, missing data, and dramatic changes in your time series.
# 
# ### Tunable Forecast
# 
# The Prophet procedure includes many possibilities for users to tweak and adjust forecasts. You can use human-interpretable parameters to improve your forecast by adding your domain knowledge.
# 
# ### Available in R & Python
# 
# We’ve implemented the Prophet procedure in R and Python, but they share the same underlying Stan code for fitting. Use whatever language you’re comfortable with to get forecasts.

# In[5]:


get_ipython().system('pip install prophet  # installation prophet')


# In[6]:


# Python
import pandas as pd
from prophet import Prophet


# In[15]:


# Python
df = pd.read_csv(r'C:\Users\Hp\Downloads\FbprophetN - Sheet1.csv')


# In[16]:


df.head()


# In[22]:


df.shape


# #####  We fit the model by instantiating a new Prophet object. Any settings to the forecasting procedure are passed into the constructor. Then you call its fit method and pass in the historical dataframe. Fitting should take 1-5 seconds.

# In[17]:


# Python
m = Prophet()


# ##### Parameters
# ----------
# growth: String 'linear', 'logistic' or 'flat' to specify a linear, logistic or
#     flat trend.
# #### changepoints: List of dates at which to include potential changepoints. If
#     not specified, potential changepoints are selected automatically.
# #### n_changepoints: Number of potential changepoints to include. Not used
#     if input `changepoints` is supplied. If `changepoints` is not supplied,
#     then n_changepoints potential changepoints are selected uniformly from
#     the first `changepoint_range` proportion of the history.
# #### changepoint_range: Proportion of history in which trend changepoints will
#     be estimated. Defaults to 0.8 for the first 80%. Not used if
#     `changepoints` is specified.
# #### yearly_seasonality: Fit yearly seasonality.
#     Can be 'auto', True, False, or a number of Fourier terms to generate.
# #### weekly_seasonality: Fit weekly seasonality.
#     Can be 'auto', True, False, or a number of Fourier terms to generate.
# #### daily_seasonality: Fit daily seasonality.
#     Can be 'auto', True, False, or a number of Fourier terms to generate.
# #### holidays: pd.DataFrame with columns holiday (string) and ds (date type)
#     and optionally columns lower_window and upper_window which specify a
#     range of days around the date to be included as holidays.
#     lower_window=-2 will include 2 days prior to the date as holidays. Also
#     optionally can have a column prior_scale specifying the prior scale for
#     that holiday.
# #### seasonality_mode: 'additive' (default) or 'multiplicative'.
# #### seasonality_prior_scale: Parameter modulating the strength of the
#     seasonality model. Larger values allow the model to fit larger seasonal
#     fluctuations, smaller values dampen the seasonality. Can be specified
#     for individual seasonalities using add_seasonality.
# #### holidays_prior_scale: Parameter modulating the strength of the holiday
#     components model, unless overridden in the holidays input.
# #### changepoint_prior_scale: Parameter modulating the flexibility of the
#     automatic changepoint selection. Large values will allow many
#     changepoints, small values will allow few changepoints.
# #### mcmc_samples: Integer, if greater than 0, will do full Bayesian inference
#     with the specified number of MCMC samples. If 0, will do MAP
#     estimation.
# #### interval_width: Float, width of the uncertainty intervals provided
#     for the forecast. If mcmc_samples=0, this will be only the uncertainty
#     in the trend using the MAP estimate of the extrapolated generative
#     model. If mcmc.samples>0, this will be integrated over all model
#     parameters, which will include uncertainty in seasonality.
# #### uncertainty_samples: Number of simulated draws used to estimate
#     uncertainty intervals. Settings this value to 0 or False will disable
#     uncertainty estimation and speed up the calculation.
# #### stan_backend: str as defined in StanBackendEnum default: None - will try to
#     iterate over all available backends and find the working one

# In[18]:


m.fit(df)


# ##### Predictions are then made on a dataframe with a column ds containing the dates for which a prediction is to be made. You can get a suitable dataframe that extends into the future a specified number of days using the helper method Prophet.make_future_dataframe. By default it will also include the dates from the history, so we will see the model fit as well.

# ##### This sets self.params to contain the fitted model parameters. It is a
# dictionary parameter names as keys and the following items:
#     k (Mx1 array): M posterior samples of the initial slope.
#     m (Mx1 array): The initial intercept.
#     delta (MxN array): The slope change at each of N changepoints.
#     beta (MxK matrix): Coefficients for K seasonality features.
#     sigma_obs (Mx1 array): Noise level.
# Note that M=1 if MAP estimation.
# 
# Parameters
# ----------
# df: pd.DataFrame containing the history. Must have columns ds (date
#     type) and y, the time series. If self.growth is 'logistic', then
#     df must also have a column cap that specifies the capacity at
#     each ds.
# kwargs: Additional arguments passed to the optimizing or sampling
#     functions in Stan.
# 
# Returns
# -------
# The fitted Prophet object.

# In[20]:


# Python
future = m.make_future_dataframe(periods=365)


# Signature: m.make_future_dataframe(periods, freq='D', include_history=True)
# Docstring:
# Simulate the trend using the extrapolated generative model.
# 
# Parameters
# ----------
# periods: Int number of periods to forecast forward.
# freq: Any valid frequency for pd.date_range, such as 'D' or 'M'.
# include_history: Boolean to include the historical dates in the data
#     frame for predictions.
# 
# Returns
# -------
# pd.Dataframe that extends forward from the end of self.history for the
# requested number of periods.

# In[21]:


future.tail()


# ##### The predict method will assign each row in future a predicted value which it names yhat. If you pass in historical dates, it will provide an in-sample fit. The forecast object here is a new dataframe that includes a column yhat with the forecast, as well as columns for components and uncertainty intervals.

# In[23]:


# Python
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# ##### You can plot the forecast by calling the Prophet.plot method and passing in your forecast dataframe.

# In[24]:


# Python
fig1 = m.plot(forecast)


# Parameters
# ----------
# ##### fcst: 
# ######  pd.DataFrame output of self.predict.
# ##### ax:
# ###### Optional matplotlib axes on which to plot.
# ##### uncertainty: 
# ###### Optional boolean to plot uncertainty intervals.
# ##### plot_cap: 
# ###### Optional boolean indicating if the capacity should be shown
#     in the figure, if available.
# ##### xlabel: 
# Optional label name on X-axis
# ##### ylabel: 
# Optional label name on Y-axis
# ##### figsize: 
# Optional tuple width, height in inches.
# ##### include_legend: 
# Optional boolean to add legend to the plot.
# 
# Returns
# -------

# ##### If you want to see the forecast components, you can use the Prophet.plot_components method. By default you’ll see the trend, yearly seasonality, and weekly seasonality of the time series. If you include holidays, you’ll see those here, too.

# In[25]:


# Python
fig2 = m.plot_components(forecast)


# ##### An interactive figure of the forecast and components can be created with plotly. You will need to install plotly 4.0 or above separately, as it will not by default be installed with prophet. You will also need to install the notebook and ipywidgets packages.

# In[26]:


# Python
from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)


# In[27]:


# Python
plot_components_plotly(m, forecast)


# ##### More details about the options available for each method are available in the docstrings, for example, via help(Prophet) or help(Prophet.fit). The R reference manual on CRAN provides a concise list of all of the available functions, each of which has a Python equivalent.
# 

# # The End
