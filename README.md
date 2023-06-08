Following are the code used in machine learning model


for regression between gameID and age

libraries used:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df=pd.read_csv('SkillCraft1_Dataset.csv')
df

df.info()

df.shape

df_describe=df.describe().T
df_describe

df.isna().sum()

df.fillna(0)

df.head(20)

df = pd.read_csv('SkillCraft1_Dataset.csv')
df_binary = df[['GameID', 'Age']]
df_binary.columns = ['GameID', 'Age']
df_binary.head(20)

data={'GameID':[52,55,56,57,58,60,61,72], 'Age': [27,23,30,19,32,27,21,17]}
df_binary=pd.DataFrame(data)
sns.lmplot(x='GameID', y='Age', data=df_binary,order=2,ci=None)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


X = np.array([52,55,56,57,58,60,61,72])
y = np.array([27,23,30,19,32,27,21,17])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

regr = LinearRegression()
regr.fit(x_train, y_train)
score = regr.score(x_test, y_test)
print('R-squared score:', score)

y_pred = regr.predict(x_test)
plt.scatter(x_test, y_test, color ='b')
plt.plot(x_test, y_pred, color ='k')
 
plt.show()

from sklearn.metrics import mean_absolute_error,mean_squared_error

mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
mse = mean_squared_error(y_true=y_test,y_pred=y_pred) 
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)





for regression between age and totalhours 

import pandas as pd
df = pd.read_csv('SkillCraft1_Dataset.csv')
df_binary = df[['Age', 'TotalHours']]
df_binary.columns = ['Age', 'TotalHours']
df_binary.head(20)

import seaborn as sns
data={'Age':[27,23,30,19,32,27,21,17], 'TotalHours': [3000,5000,200,400,500,70,240,10000]}
df_binary=pd.DataFrame(data)
sns.lmplot(x='Age', y='TotalHours', data=df_binary,order=2,ci=None)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


X = np.array([27,23,30,19,32,27,21,17,20,18,16])
y = np.array([3000,5000,200,400,500,70,240,10000,2708,800,6000])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

regr = LinearRegression()
regr.fit(x_train, y_train)
score = regr.score(x_test, y_test)
print('R-squared score:', score)

import matplotlib.pyplot as plt
y_pred = regr.predict(x_test)
plt.scatter(x_test, y_test, color ='b')
plt.plot(x_test, y_pred, color ='k')
 
plt.show()

from sklearn.metrics import mean_absolute_error,mean_squared_error

mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
mse = mean_squared_error(y_true=y_test,y_pred=y_pred) 
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
