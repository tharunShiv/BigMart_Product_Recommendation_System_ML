# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# how to determine the min_support?
# consider different reasonable values of the rules and check result and make changes if necessary
# consider that you want to have product is puchased atleast 3 times a day
# so that is 7*3 = 21 times a week
# so support for this is
# 21/total number of Transactions
# 21/7500 = 0.003
# hence this is chosen as the min_support

# min_length => min_number of products in a rule which we get in the result

# how to determine the min_confidence?
# mostly 0.8 is min_confidence, but it doesnt work with that
# consider 0.2 as min_confidence
# this is chosen after analysing the datasetself.

# minlift?
# we have decided to use lift of 4, 5, 6
# so 3
# but this may depend on the dataset
# so spend more time on these parameters


# Visualising the results
# the rules here are already sorted according to
# its own relevence
# based on support, confidence, rules
# so we are not sorting it manually

results = list(rules)

# in the results, index 0 is the most relevant rule
# in the index 0
# 2nd row is the support
# now, double click 3rd row, again click on the value
# to get further details
# here index 2 is confidence
# index 3 is lift
