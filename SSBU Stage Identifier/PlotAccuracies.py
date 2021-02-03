########################################################################################################################
# Plots accuracies from .csv file made during NN Training.                                                             #
########################################################################################################################


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('SSBU_hyperparam.csv')
ax = plt.gca()
columns = df.columns

df.plot(x=columns, y=columns[1:len(columns)], ax=ax)
plt.savefig('acc.png')
plt.show()



