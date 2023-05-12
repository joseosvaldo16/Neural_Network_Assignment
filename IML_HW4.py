# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sys
import sklearn
from sklearn import metrics, neural_network
import itertools
import Utility
import time
import scipy
from sklearn import preprocessing, naive_bayes
import math 
plt.rcParams['figure.dpi'] = 150
import scipy.stats as stats


sklearn.__version__
##Jose Vera (jvera3@hawk.iit.edu)
##Homework 4

# %%
df = pd.read_csv('Purchase_Likelihood.csv').dropna()
dfx = df.drop(columns='insurance')

catName = ['group_size','homeowner','married_couple']

# %%
def RowWithColumn (
   rowVar,          # Row variable
   columnVar,       # Column predictor
   show = 'ROW'):   # Show ROW fraction, COLUMN fraction, or BOTH table

   countTable = pd.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
   print("Frequency Table: \n", countTable)
   print( )

   if (show == 'ROW' or show == 'BOTH'):
       rowFraction = countTable.div(countTable.sum(1), axis='index')
       print("Row Fraction Table: \n", rowFraction)
       print( )

   if (show == 'COLUMN' or show == 'BOTH'):
       columnFraction = countTable.div(countTable.sum(0), axis='columns')
       print("Column Fraction Table: \n", columnFraction)
       print( )

   return

# %%
catGroupSize = df['group_size'].unique()
catHomeowner = df['homeowner'].unique()
catMarriedCouple = df['married_couple'].unique()
catInsurance = df['insurance'].unique()

RowWithColumn(rowVar = df['insurance'], columnVar = df['group_size'], show = 'ROW')
print('-----------')
RowWithColumn(rowVar = df['insurance'], columnVar = df['homeowner'], show = 'ROW')
print('-----------')
RowWithColumn(rowVar = df['insurance'], columnVar = df['married_couple'], show = 'ROW')


# %%
df = df.astype('category')
xTrain = pd.get_dummies(df[['group_size','homeowner','married_couple']])
yTrain = df.insurance

# %%
# Correctly Use sklearn.naive_bayes.CategoricalNB
feature = catName

labelEnc = preprocessing.LabelEncoder()
yTrain = labelEnc.fit_transform(df['insurance'])
yLabel = labelEnc.inverse_transform([0, 1])

uGropuSize = np.unique(df['group_size'])
uHomeowner = np.unique(df['homeowner'])
uMarriedCouple = np.unique(df['married_couple'])

featureCategory = [uGropuSize, uHomeowner, uMarriedCouple]
print(featureCategory)

featureEnc = preprocessing.OrdinalEncoder(categories = featureCategory)
xTrain = featureEnc.fit_transform(df[['group_size', 'homeowner', 'married_couple']])

_objNB = naive_bayes.CategoricalNB(alpha = 0)
thisModel = _objNB.fit(xTrain, yTrain)

# %%
print('Number of samples encountered for each class during fitting')
print(yLabel)
print(_objNB.class_count_)
print('\n')
 
print('Probability of each class:')
print(yLabel)
print(np.exp(_objNB.class_log_prior_))
print('\n')

print('Number of samples encountered for each (class, feature) during fitting')
for i in range(3):
   print('Feature: ', feature[i])
   print(featureCategory[i])
   print(_objNB.category_count_[i])
   print('\n')

print('Empirical probability of features given a class, P(x_i|y)')
for i in range(3):
   print('Feature: ', feature[i])
   print(featureCategory[i])
   print(np.exp(_objNB.feature_log_prob_[i]))
   print('\n')

# %%
dataGropu = _objNB.category_count_[0]
dataHome = _objNB.category_count_[1]
dataMarried = _objNB.category_count_[2]

# %%
#Function that helps calculate cramerV statistic

def cramerV(d):
    X2 = stats.chi2_contingency(d, correction=False)[0]
    N = np.sum(d)
    minimum_dimension = min(d.shape)-1
  
    # Calculate Cramer's V
    result = np.sqrt((X2/N) / minimum_dimension)
    return result


# %%
print(cramerV(dataGropu))
print(cramerV(dataHome))
print(cramerV(dataMarried))

# %%


# %%


# %%

df2 = pd.read_excel('Homeowner_Claim_History.xlsx').replace(to_replace='None', value=np.nan).dropna()
#Category Names
catName = ['f_primary_age_tier','f_primary_gender','f_marital','f_residence_location','f_fire_alarm_type','f_mile_fire_station','f_aoi_tier']
#target
yName = 'frequency'
##Calculate the frequency and add column to data frame for training
trainData = df2.assign(frequency=lambda x: x.num_claims/ df2.exposure).replace(to_replace='None', value=np.nan).dropna()
n_sample = trainData.shape[0]


# %%
# Reorder the categories of the target variables in descending frequency
u = trainData[yName].astype('category').copy()

u_freq = u.value_counts(ascending = False)
trainData[yName] = u.cat.reorder_categories(list(u_freq.index)).copy()

# Reorder the categories of the categorical variables in ascending frequency
for pred in catName:
    u = trainData[pred].astype('category').copy()
    u_freq = u.value_counts(ascending = True)
    trainData[pred] = u.cat.reorder_categories(list(u_freq.index)).copy()

# %%
X = pd.get_dummies(trainData[catName].astype('category'))
X.insert(0, '_BIAS_', 1.0)

# Identify the aliased parameters
n_param = X.shape[1]
XtX = X.transpose().dot(X)
origDiag = np.diag(XtX)
XtXGinv, aliasParam, nonAliasParam = Utility.SWEEPOperator (n_param, XtX, origDiag, sweepCol = range(n_param), tol = 1.0e-7)
X_reduce = X.iloc[:, list(nonAliasParam)].drop(columns = ['_BIAS_']).replace(to_replace='None', value=np.nan).dropna()

y = pd.Series(trainData[yName].values).replace(to_replace='None', value=np.nan).dropna()


# %%
# Grid Search for the best neural network architecture
actFunc = ['identity','tanh']
nLayer = range(1,11,1)
nHiddenNeuron = range(1,6,1)
combList = itertools.product(actFunc, nLayer, nHiddenNeuron)

result_list = []

# %%
for comb in combList:
   time_begin = time.time()
   actFunc = comb[0]
   nLayer = comb[1]
   #Interate through different combinations of the model and save results into results_list
   nHiddenNeuron = comb[2]
   #Create and train MLPREgressor model
   nnObj = neural_network.MLPRegressor(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
              activation =  actFunc, verbose = False, max_iter = 10000, random_state = 31010)
   #Fit data on to model
   thisFit = nnObj.fit(X_reduce, y)
   #Save predicted values
   y_predProb = pd.DataFrame(nnObj.predict(X_reduce))

   nIter = nnObj.n_iter_
   blv = nnObj.best_loss_
   #Calculate the RMSE
   rase = metrics.mean_squared_error(y, y_predProb)
   yMean = y.astype(float).mean()
   #Calculate the Relative Error
   rError = ((pd.DataFrame(y).values - y_predProb)**2).sum()[0]/((pd.DataFrame(y).values-yMean)**2).sum()
   #Calculate the person correlation
   pearson, _ = scipy.stats.pearsonr(y, y_predProb)
   #Calculate time elapsed
   elapsed_time = time.time() - time_begin
   result_list.append([actFunc, nLayer, nHiddenNeuron, nIter, blv, rase, rError, pearson, elapsed_time])


# %%
result_df = pd.DataFrame(result_list, columns = ['Activation Function', 'nLayer', 'nHiddenNeuron', 'nIter', 'blv','RMSE', 'rError','pearson', 'Elapsed Time'])

# %%


# %%
result_df.sort_values(by='RMSE', ascending= True)

# %%
nnObj = neural_network.MLPRegressor(hidden_layer_sizes = (5,)*3,
              activation = 'tanh', verbose = False, max_iter = 10000, random_state = 31010)
thisFit = nnObj.fit(X_reduce, y)
y_predProb = pd.DataFrame(nnObj.predict(X_reduce))
observed = pd.DataFrame(y).astype(float).copy()

# %%
y_predProb.max()

# %%
observed.max()

# %%
plt.plot(observed,y_predProb, 'ro', markersize = 3 )
plt.ylabel('Predicted')
plt.xlabel('Observed')
plt.title('Predicted vs Observed Frequency')
plt.show()


# %%
sResidual = observed - y_predProb

pResidual = sResidual/ ((y_predProb)**(0.5))

# %%

plt.plot(observed,sResidual, 'ro',label='sResidual', markersize = 3)
plt.plot(observed,pResidual,'ro', color = 'blue', label='pResidual', markersize = 3)
plt.ylabel('Predicted')
plt.xlabel('Observed')
plt.title('Residual vs Observed')
plt.xticks(np.arange(0, 201, 20))
plt.legend()
plt.show()


