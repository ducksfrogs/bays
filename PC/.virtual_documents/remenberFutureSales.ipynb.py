get_ipython().getoutput("ls ../data/")


import numpy as np
import pandas as pd
import random as rd

import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings("ignore")


sales = pd.read_csv("../data/sales_train.csv")

item_cat = pd.read_csv("../data/item_categories.csv")
sub = pd.read_csv("../data/items.csv")
sub = pd.read_csv("../data/sample_submission.csv")
shops = pd.read_csv("../data/shops.csv")
test = pd.read_csv("../data/test.csv")


sales


sales.date = sales.date.apply(lambda x:datetime.datetime.strptime(x, 'get_ipython().run_line_magic("d.%m.%Y'))", "")


sales.info()


sales



