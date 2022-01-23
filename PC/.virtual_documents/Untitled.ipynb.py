import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('data.csv')


data


sum(data['amount'])


data['amount'].mean()


data.describe()


data_sorted = data.sort_values('amount')


data_wiht_mean =data['amount'==58] = data['amount'].mean()


data_wiht_mean














data_wiht_mean = data.replace(58, data['amount'].mean())


data_wiht_mean


data_wiht_mean.describe()


data_sorted


data_trimed = data_sorted[0:7]


data_trimed.describe()



