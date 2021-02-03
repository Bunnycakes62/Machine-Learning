# Time series Monte Carlo Simulation for the spread of Coronavirus

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

import os
# print(os.listdir("Dataset"))

data = pd.read_csv("Dataset/2019_nC0v_20200121_20200126_cleaned.csv")
data = data.sort_values(by="Date last updated")

data = data[data.Country == 'Mainland China'].iloc[1:]



confirmed = np.log(1+np.abs(data.Confirmed))
_mean = confirmed.mean()
_var = confirmed.var()
drift = _mean - (0.5*_var)
stddev = confirmed.std()

t_interval = 4
itr = 10 # number of different forecasts

daily_confirmed = np.exp(drift + stddev * norm.pdf(np.random.rand(t_interval, itr)))


# P = P0 * exp(rate * time)
P0 = data.Confirmed.iloc[-1]

confirmed_list = np.zeros_like(daily_confirmed)
confirmed_list[0] = P0

# forecast for 30 days
for t in range(1, t_interval):
    confirmed_list[t] = confirmed_list[t-1] * daily_confirmed[t]
confirmed_list = pd.DataFrame(confirmed_list)
confirmed_list['Confirmed'] = confirmed_list[0]

confirm = data.Confirmed
confirm = pd.DataFrame(confirm)
frames = [confirm, confirmed_list]
monte_carlo_forecast = pd.concat(frames, sort=True)

monte_carlo = monte_carlo_forecast.iloc[:,:].values
plt.plot(monte_carlo)

