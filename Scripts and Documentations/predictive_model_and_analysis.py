# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 08:07:14 2024

@author: keneo
"""

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the dataset
housing = pd.read_csv(r"C:\Users\keneo\OneDrive\Documents\Scripting in Data Analysis Coursework 2\UK_Housing_Data.csv")
 
 
# (A.) Identify and  address any issues in the dataset. Then conduct exploratory data analysis on the dataset
 
# check the dataset info
housing_info = housing.info()
 
# correct the datatype of 'TDate' column
housing['TDate'] = pd.to_datetime(housing['TDate'])

# check for missing values in the dataset
housing_missing_values = housing.isnull().sum()

# drop unamed column and columns with missing values
housing.drop(columns = ['Unnamed: 0', 'TID', 'PAON', 'Town/City', 'District', 'County', 'SAON', 'Street', 'Locality', 'RecordStatus'], inplace = True)

# drop the missing rows in the 'PostCode' column
housing.dropna(subset = ['PostCode'], inplace = True)

# correct the values in 'PType' column
housing['PType'].replace({'D':'Detached',
                          'S':'Semi-Detached',
                          'T':'Terraced',
                          'F':'Flats',
                          'O':'Other'}, inplace = True)

# correct the values in 'Old/New' column
housing['Old/New'].replace({'Y':'New',
                            'N':'Old'}, inplace = True)

# correct the values in 'Duration' column
housing['Duration'].replace({'F':'Freehold',
                             'L':'Leasehold'}, inplace = True)

# correct the values in 'PPD_Type' column
housing['PPD_Type'].replace({'A':'Standard Price Paid',
                             'B':'Additional Price Paid'}, inplace = True)

# check for duplicated values in the dataset
housing_duplicated_values = housing.duplicated().sum()
 
 
# (B.) Analyse the relationships between price and the other features.
 
# create a list of columns to encode
encoded_columns = ['PostCode', 'PType', 'Old/New', 'Duration', 'PPD_Type']
 
# create a dataframe to store the encoded values
encoded_housing = pd.DataFrame()
 
# assign the target encoder function to a variable
target_encoder = TargetEncoder()
 
# create a loop to encode each column in the list and store them in the dataframe
for col in encoded_columns:         
    encoded_values = target_encoder.fit_transform(housing[col], housing['Price'])   
    encoded_housing[col] = encoded_values
 
# add 'TDate' and 'Price' columns from 'housing' dataframe to 'encoded_housing' dataframe
encoded_housing['TDate'] = housing['TDate']
encoded_housing['Price'] = housing['Price']
 
# find the correlation
housing_corr = encoded_housing.corr()['Price']
print('\nRelationship between price and other features: \n',housing_corr)
 
 
# (C.) Analyse and visualise the impact of Brexit on house prices.
 
# assign brexit variable to the brexit date
brexit = pd.to_datetime("2020-01-31")
 
# separate housing data into before and after brexit
before_brexit = housing.query('TDate < @brexit')
during_brexit = housing.query('TDate == @brexit')
after_brexit = housing.query('TDate > @brexit')
 
# get the average house price before and after brexit
ave_price_before_brexit = before_brexit['Price'].mean()
avg_price_during_brexit = during_brexit['Price'].mean()
avg_price_after_brexit = after_brexit['Price'].mean()
 
# plot the chat
plt.figure(figsize=(15, 5))
plt.bar(['BEFORE BREXIT', 'DURING BREXIT', 'AFTER BREXIT'], [ave_price_before_brexit, avg_price_during_brexit, 
                                                             avg_price_after_brexit], color=['red', 'green', 'yellow'])
plt.title('AVERAGE HOUSE PRICES BEFORE, DURING, AND AFTER BREXIT')
plt.ylabel('AVERAGE HOUSE PRICES')
plt.show()
 
 
# (D.) Build a predictive model to estimate house prices.
 
# select the features
X = encoded_housing[['PostCode', 'PType']]
y = housing['Price']
 
# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
 
 
########## Build a Linear Regression  Model ##########
# train the model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# make the prediction
y_pred_lin = lin_reg.predict(X_test)

# calculate evaluation metrics
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
r2_lin = r2_score(y_test, y_pred_lin)
 
# print the result
print("Root Mean Squared Error for Linear Regression  Model:", rmse_lin)
print("R2 Score for Linear Regression  Model:", r2_lin)
 
 
########## Build a KNeighbors Regressor Model ##########
# train the model
kn_reg = KNeighborsRegressor()
kn_reg.fit(X_train, y_train)

# make the predictions
y_pred_kn = kn_reg.predict(X_test)

# calculate evaluation metrics
rmse_kn = np.sqrt(mean_squared_error(y_test, y_pred_kn))
r2_kn = r2_score(y_test, y_pred_kn)
 
# print the result
print("Root Mean Squared Error for KNeighbors Regressor Model:", rmse_kn)
print("R2 Score for KNeighbors Regressor Model:", r2_kn)
 
 
########## Build SGD Regressor Model #########
# train the model
sgd_reg = make_pipeline(StandardScaler(), SGDRegressor())
sgd_reg.fit(X_train, y_train)
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('sgdregressor', SGDRegressor())])
 
# make the predictions
y_pred_sgd = sgd_reg.predict(X_test)

# calculate evaluation metrics
rmse_sgd = np.sqrt(mean_squared_error(y_test, y_pred_sgd))
r2_sgd = r2_score(y_test, y_pred_sgd)
 
# print the result
print("Root Mean Squared Error for SGD Regressor Model:", rmse_sgd)
print("R2 Score for SGD Regressor Model:", r2_sgd)
 

# visualize using scatterplot
# plot the chat
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (15,6))

########## Linear Regression Model ##########
ax1.scatter(y_test, y_pred_lin, color=['r'], label='Predicted')
ax1.scatter(y_test, y_test, color=['b'], label='Actual')
ax1.set_title('Linear Regression Model')
ax1.legend()

########## KNeighbors Regression Model ##########
ax2.scatter(y_test, y_pred_kn, color=['b'], label='Predicted')
ax2.scatter(y_test, y_test, color=['g'], label='Actual')
ax2.set_title('KNeighbors Regression Model')
ax2.legend()

########## SGD Regression Model ##########
ax3.scatter(y_test, y_pred_sgd, color=['g'], label='Predicted')
ax3.scatter(y_test, y_test, color=['r'], label='Actual')
ax3.set_title('SGD Regression Model')
ax3.legend()






