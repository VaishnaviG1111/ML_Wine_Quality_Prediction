# -*- coding: utf-8 -*-
"""
Created on Sun May 19 23:32:06 2024

@author: HP
"""
#Wine Quality Predictions 

#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Import dataset
data = pd.read_csv(r'C:\Users\HP\Desktop\All Project\Wine Quality files\wineQualityReds.csv')


# dividing the dataset into dependent and independent variables
x = data.iloc[:,:-1]
y = data.iloc[:,11]

#Splitting dataset into train_set and test_set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 44)

# determining the shapes of training and testing sets
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

## standard scaling 
#from sklearn.preprocessing import StandardScaler
#
#sc = StandardScaler()
#x_train = sc.fit_transform(x_train)
#x_test = sc.fit_transform(x_test)
#Applying MLR


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


#Building an optimal model using backward elimination
import statsmodels.api as sm
x = np.append(arr= np.ones((1599,1)).astype(int) , values= x, axis = 1 )
x_opt = x[: ,[0,1,2,3,4,5,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#Building an optimal model using backward elimination
import statsmodels.api as sm
x = np.append(arr= np.ones((1599,1)).astype(int) , values= x, axis = 1 )
x_opt = x[: ,[0,1,2,4,5,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()


#Building an optimal model using backward elimination
import statsmodels.api as sm
x = np.append(arr= np.ones((1599,1)).astype(int) , values= x, axis = 1 )
x_opt = x[: ,[0,1,2,4,5,6,7,8,9]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#Building an optimal model using backward elimination
import statsmodels.api as sm
x = np.append(arr= np.ones((1599,1)).astype(int) , values= x, axis = 1 )
x_opt = x[: ,[0,1,2,4,5,7,8,9]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#Building an optimal model using backward elimination
import statsmodels.api as sm
x = np.append(arr= np.ones((1599,1)).astype(int) , values= x, axis = 1 )
x_opt = x[: ,[0,1,2,5,7,8,9]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#Building an optimal model using backward elimination
import statsmodels.api as sm
x = np.append(arr= np.ones((1599,1)).astype(int) , values= x, axis = 1 )
x_opt = x[: ,[0,1,2,5,7,9]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#Building an optimal model using backward elimination
import statsmodels.api as sm
x = np.append(arr= np.ones((1599,1)).astype(int) , values= x, axis = 1 )
x_opt = x[: ,[0,1,2,5,9]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()


#data.head()
#data.columns
#data.describe()
#data.info()
#data.isnull().sum()
data['quality'].value_counts()

data['quality'] = data['quality'].map({3 : 'bad', 4 :'bad', 5: 'bad',
                                      6: 'good', 7: 'good', 8: 'good'})
#Encoding bad/goo/normal 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['quality'] = le.fit_transform(data['quality'])
data['quality'].value_counts
sns.countplot(data['quality'])
#Applying Logistic Regression

from sklearn.linear_model import LogisticRegression
LR= LogisticRegression()
LR.fit(x_train, y_train)

y_pred = LR.predict(x_test)

# calculating the training and testing accuracies
print("Training accuracy :", LR.score(x_train, y_train))
print("Testing accuracy :", LR.score(x_test, y_test))

#Making Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm_LR=confusion_matrix(y_test,y_pred)

#Applying DecisionTree 
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(x_train,y_train)

y_pred = DT.predict(x_test)

# calculating the training and testing accuracies
print("Training accuracy :", DT.score(x_train, y_train))
print("Testing accuracy :", DT.score(x_test, y_test))

#Making Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm_DT=confusion_matrix(y_test,y_pred)

#Importingg Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RF= RandomForestClassifier(n_estimators = 200)
RF.fit(x_train, y_train)

y_pred = RF.predict(x_test)

# calculating the training and testing accuracies
print("Training accuracy :", model.score(x_train, y_train))
print("Testing accuracy :", model.score(x_test, y_test))

#Making Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm_RF=confusion_matrix(y_test,y_pred)
