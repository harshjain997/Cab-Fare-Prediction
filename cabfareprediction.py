#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import os #getting access to input files
import pandas as pd # Importing pandas for performing EDA
import numpy as np  # Importing numpy for Linear Algebric operations
import matplotlib.pyplot as plt # Importing for Data Visualization
import seaborn as sns # Importing for Data Visualization
from sklearn.linear_model import LinearRegression #ML algorithm
from sklearn.model_selection import train_test_split #splitting dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pprint import pprint
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt # Importing for Data Visualization
import seaborn as sns # Importing for Data Visualization
from collections import Counter 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[2]:


os.chdir(r"C:\Users\harsh\Desktop\harsh jain")
print(os.getcwd())


# # loading data

# In[3]:


train = pd.read_csv("train_cab.csv",na_values={"pickup_datetime":"43"})
test = pd.read_csv("test.csv")


# # data understanding

# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


train.shape  #for finding rows and column


# In[7]:


test.shape  #for finding rows and column


# In[8]:


print(train.dtypes)
print("-----------------------------------------")
print(test.dtypes)


# here data type of amount in train and datetime is of object type so we need to change these to particular formats

# # First we need to clean our data and analyse if there are any missing values or not

# In[9]:


train.describe() #describe function describe all statics of data


# In[10]:


test.describe()


# # changing data types to correct format

# In[11]:


train["fare_amount"] = pd.to_numeric(train["fare_amount"],errors = "coerce")
train['pickup_datetime'] =  pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"],format= "%Y-%m-%d %H:%M:%S UTC")


# In[12]:


print(train.dtypes)
print("---------------------------------")
print(test.dtypes)


# In[13]:


train.dropna(subset= ["pickup_datetime"]) #dropping null value 


# In[14]:


# putting year, month , date , day , hour and min of both test and train data set in different column
train['year'] = train['pickup_datetime'].dt.year
train['Month'] = train['pickup_datetime'].dt.month
train['Date'] = train['pickup_datetime'].dt.day
train['Day'] = train['pickup_datetime'].dt.dayofweek
train['Hour'] = train['pickup_datetime'].dt.hour
train['Minute'] = train['pickup_datetime'].dt.minute


# In[15]:


test['year'] = test['pickup_datetime'].dt.year
test['Month'] = test['pickup_datetime'].dt.month
test['Date'] = test['pickup_datetime'].dt.day
test['Day'] = test['pickup_datetime'].dt.dayofweek
test['Hour'] = test['pickup_datetime'].dt.hour
test['Minute'] = test['pickup_datetime'].dt.minute


# In[16]:


# rechecking data tyes 
print(train.dtypes)
print("-----------------------------------------")
print(test.dtypes)


# In[17]:


# removing null value from datetime 
train = train.drop(train[train['pickup_datetime'].isnull()].index, axis=0)
print(train.shape)
print(train['pickup_datetime'].isnull().sum())


# Observations :
# An outlier in pickup_datetime column of value 43
# 
# Passenger count should not exceed 6(even if we consider SUV)
# 
# Latitudes range from -90 to 90. Longitudes range from -180 to 180
# 
# Few missing values and High values of fare and Passenger count are present. So, decided to remove them.
# Checking the Datetime Variable 

# In[18]:


train["passenger_count"].describe()


# maximum number of passanger count is 5345 which is actually not possible. So reducing the passenger count to 6 (even if we consider the SUV)
# and also removing passenger count with value 0.

# In[19]:


train = train.drop(train[train["passenger_count"]> 6 ].index, axis=0)
train = train.drop(train[train["passenger_count"] == 0 ].index, axis=0)


# In[20]:


train["passenger_count"].describe()


# In[21]:


train["passenger_count"].sort_values(ascending= True)


# In[22]:


#we have to remove passanger_count missing values rows
train = train.drop(train[train['passenger_count'].isnull()].index, axis=0)
print(train.shape)
print(train['passenger_count'].isnull().sum())


# we aslo have to remove fraction values from passenger_count i.e. 0.12 because count should be 1 or more then 1

# In[23]:


train = train.drop(train[train["passenger_count"] == 0.12 ].index, axis=0)
train.shape


# In[24]:


# we will put fare_amount in decending order to know whether the outliers are present or not
train["fare_amount"].sort_values(ascending=False)


# In[25]:


Counter(train["fare_amount"]<0)


# In[26]:


train = train.drop(train[train["fare_amount"]<0].index, axis=0)
train.shape


# checking if there anynegative value in fare or not

# In[27]:


train["fare_amount"].min()


# In[28]:


#Also remove the row where fare amount is zero
train = train.drop(train[train["fare_amount"]<1].index, axis=0)
train.shape


# In[29]:


#decending order of fare amount helped us to find the outlier value i.e. 454 so we will remove the rows having fare_amount more than 454.
train = train.drop(train[train["fare_amount"]> 454 ].index, axis=0)
train.shape


# In[30]:


# we will also remove rows which is having missing Fare_amount
train = train.drop(train[train['fare_amount'].isnull()].index, axis=0)
print(train.shape)
print(train['fare_amount'].isnull().sum())


# no we have to  check pickup  latitude and longitude and drop_off latitude and longitude

# In[31]:


#Lattitude----(-90 to 90) and Longitude----(-180 to 180) dropping other values which doesn't occure in this range

train[train['pickup_latitude']<-90]
train[train['pickup_latitude']>90]


# In[32]:


train[train['pickup_longitude']<-180]
train[train['pickup_longitude']>180]


# In[33]:


train[train['dropoff_longitude']<-180]
train[train['dropoff_longitude']>180]


# In[34]:


train[train['dropoff_latitude']<-90]
train[train['dropoff_latitude']>90]


# In[35]:


train.shape


# In[36]:


train.isnull().sum() 


# In[37]:


test.isnull().sum()


# now the data cleaning is done , now we need to calculate the distance using latitude and longitude points so that we can proceed further.
# # we can calculate distance using haversian formula
# 
# import math
# 
# def distance(origin, destination):
#     lat1, lon1 = origin
#     lat2, lon2 = destination
#     radius = 6371 # km
# 
#     dlat = math.radians(lat2-lat1)
#     dlon = math.radians(lon2-lon1)
#     a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
#         * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
#     d = radius * c
# 
#     return d

# In[38]:


from math import *

def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km


# In[39]:


train['distance'] = train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[40]:


test['distance'] = test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[41]:


train.head() #we added distance calculated column 


# In[42]:


test.head()


# In[43]:


#nunique() method is used to get number of all unique values
train.nunique()


# In[44]:


test.nunique()


# # now that we calculated distance we can remove outliers record of ditance also

# In[45]:


d = train['distance'].sort_values(ascending=False)
d.tail(30)


# In[46]:


Counter(train['distance'] == 0)


# In[47]:


Counter(test['distance'] == 0)


# In[48]:


#removing data with distance 0
train = train.drop(train[train['distance']== 0].index, axis=0)
train.shape


# In[49]:


test = test.drop(test[test['distance']== 0].index, axis=0)
test.shape


# # we willl also remove outlier i.e. distance above 130kms 

# In[50]:


train = train.drop(train[train['distance'] > 130 ].index, axis=0)
train.shape


# In[51]:


train.head()


# # now removing unneccessary columns from our train dataset

# In[52]:


drop = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'Minute']
train = train.drop(drop, axis = 1)


# In[53]:


train.head()


# In[54]:


train.dtypes


# In[55]:


#changing datatypes
train['passenger_count'] = train['passenger_count'].astype('int64')
train['year'] = train['year'].astype('int64')
train['Month'] = train['Month'].astype('int64')
train['Date'] = train['Date'].astype('int64')
train['Day'] = train['Day'].astype('int64')
train['Hour'] = train['Hour'].astype('int64')


# In[56]:


train.dtypes


# In[57]:


#doing same for test data set
drop_test = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'Minute']
test = test.drop(drop_test, axis = 1)


# In[58]:


test.head()


# # now moving towards visualization to get good insight of data

# Visualization can give insight to:
# 
# Passengers_count effects the the fare
# 
# date and time effects the fare
# 
# Will also tell does day of the week effects the fare
# 
# Distance effects the fare

# In[59]:


# Count plot on passenger count
plt.figure(figsize=(10,5))
sns.countplot(x="passenger_count", data=train)


# In[60]:


#Relationship beetween number of passengers and Fare
plt.figure(figsize=(10,5))
plt.scatter(x=train['passenger_count'], y=train['fare_amount'], s=10)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
plt.show()


# In[61]:


#visualisation between date and Fare
plt.figure(figsize=(10,5))
plt.scatter(x=train['Date'], y=train['fare_amount'], s=10)
plt.xlabel('Date')
plt.ylabel('Fare')
plt.show()


# In[62]:


plt.figure(figsize=(10,5))
train.groupby(train["Hour"])['Hour'].count().plot(kind="bar")
plt.show()


# In[63]:


#between Time and Fare
plt.figure(figsize=(10,5))
plt.scatter(x=train['Hour'], y=train['fare_amount'], s=10)
plt.xlabel('Hour')
plt.ylabel('Fare')
plt.show()


# Observations :
# By seeing the above plots we can easily conclude that:
# 
# passenger_count 1 are most frequent travellers and highest Fare are coming from passenger_count 1 and 2.
# 
# Lowest cabs at 5 AM and highest around 7 PM i.e the office rush hours
# 
# We can observe that the cabs taken at 7 am and 23 Pm are the costliest i.e. that cabs taken early in morning and late at night are costliest

# In[64]:


#impact of Day on the number of cab rides
plt.figure(figsize=(10,5))
sns.countplot(x="Day", data=train)


# In[65]:


#between day and Fare
plt.figure(figsize=(10,5))
plt.scatter(x=train['Day'], y=train['fare_amount'], s=10)
plt.xlabel('Day')
plt.ylabel('Fare')
plt.show()


# In[66]:


#Relationship between distance and fare 
plt.figure(figsize=(10,5))
plt.scatter(x = train['distance'],y = train['fare_amount'])
plt.xlabel('Distance')
plt.ylabel('Fare')
plt.show()


# Other observations :
# The day of the week does not affect the number of cabs ride
# 
# and obiviously distance will effect the amount of fare

# # Feature Scaling :

# In[67]:


#Normality check of training data  whether the data is uniformly distributed or not-

for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# In[68]:


#since skewness of target variable is high, apply log transform to reduce the skewness-
train['fare_amount'] = np.log1p(train['fare_amount'])

#since skewness of distance variable is high, apply log transform to reduce the skewness-
train['distance'] = np.log1p(train['distance'])


# In[69]:


for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# Here we can see bell shaped distribution. Hence our continous variables are now normally distributed, we will use not use any Feature Scalling technique. i.e, Normalization or Standarization for our training data

# In[70]:


# same thing for test data set
sns.distplot(test['distance'],bins='auto')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# In[71]:


#since skewness of distance variable is high, apply log transform to reduce the skewness-
test['distance'] = np.log1p(test['distance'])


# In[72]:


sns.distplot(test['distance'],bins='auto')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# # Now we are done with all the things i.e. data cleaning feature scaling of data its time to apply machine learning algorithm

# In[73]:


#For further modelling we will apply train test split
X_train, X_test, y_train, y_test = train_test_split( train.iloc[:, train.columns != 'fare_amount'], 
                         train.iloc[:, 0], test_size = 0.20, random_state = 1)


# In[74]:


print(X_train.shape)
print(X_test.shape)


# # # starting with Linear Regression Model :

# In[75]:


# Building model on top of training dataset
fit_LR = LinearRegression().fit(X_train , y_train)

pred_train_LR = fit_LR.predict(X_train)#prediction on train data


pred_test_LR = fit_LR.predict(X_test)#prediction on test data


# In[76]:


##calculating RMSE for test data
RMSE_test_LR = np.sqrt(mean_squared_error(y_test, pred_test_LR))
print("RMSE for Test data = "+str(RMSE_test_LR))

##calculating RMSE for train data
RMSE_train_LR= np.sqrt(mean_squared_error(y_train, pred_train_LR))
print("RMSE for Training data = "+str(RMSE_train_LR))


# In[77]:


#calculate R^2 for train data
from sklearn.metrics import r2_score
r2_score(y_train, pred_train_LR)


# In[78]:


r2_score(y_test, pred_test_LR)


# # Decision tree Model :

# In[79]:


fit_DT = DecisionTreeRegressor(max_depth = 2).fit(X_train,y_train)


# In[80]:


#prediction on train data
pred_train_DT = fit_DT.predict(X_train)

#prediction on test data
pred_test_DT = fit_DT.predict(X_test)


# In[81]:


##calculating RMSE for train data
RMSE_train_DT = np.sqrt(mean_squared_error(y_train, pred_train_DT))
print("RMSE for Training data = "+str(RMSE_train_DT))
##calculating RMSE for test data
RMSE_test_DT = np.sqrt(mean_squared_error(y_test, pred_test_DT))
print("RMSE for Test data = "+str(RMSE_test_DT))


# In[82]:


print(r2_score(y_train, pred_train_DT))
print(r2_score(y_test, pred_test_DT))


# # Random Forest Model :

# In[83]:


fit_RF = RandomForestRegressor(n_estimators = 200).fit(X_train,y_train)


# In[84]:


#prediction on train data
pred_train_RF = fit_RF.predict(X_train)
#prediction on test data
pred_test_RF = fit_RF.predict(X_test)


# In[85]:


##calculating RMSE for train data
RMSE_train_RF = np.sqrt(mean_squared_error(y_train, pred_train_RF))
print("RMSE for Training data = "+str(RMSE_train_RF))
##calculating RMSE for test data
RMSE_test_RF = np.sqrt(mean_squared_error(y_test, pred_test_RF))
print("RMSE for Test data = "+str(RMSE_test_RF))


# In[86]:


print(r2_score(y_train, pred_train_RF)) #train
print(r2_score(y_test, pred_test_RF)) #test


# # Gradient Boosting :

# In[87]:


fit_GB = GradientBoostingRegressor().fit(X_train, y_train)


# In[88]:


#prediction on train data
pred_train_GB = fit_GB.predict(X_train)

#prediction on test data
pred_test_GB = fit_GB.predict(X_test)


# In[89]:


##calculating RMSE for train data
RMSE_train_GB = np.sqrt(mean_squared_error(y_train, pred_train_GB))
print("RMSE for Training data = "+str(RMSE_train_GB))

##calculating RMSE for test data
RMSE_test_GB = np.sqrt(mean_squared_error(y_test, pred_test_GB))
print("RMSE for Test data = "+str(RMSE_test_GB))


# In[90]:


print(r2_score(y_test, pred_test_GB))
#calculate R^2 for train data
r2_score(y_train, pred_train_GB)


# # Optimizing the results with parameters tuning :

# In[91]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[92]:


from sklearn.model_selection import train_test_split,RandomizedSearchCV


# In[93]:


##Random Search CV on Random Forest Model

RRF = RandomForestRegressor(random_state = 0)
n_estimator = list(range(1,20,2))
depth = list(range(1,100,2))

# Create the random grid
rand_grid = {'n_estimators': n_estimator,
               'max_depth': depth}

randomcv_rf = RandomizedSearchCV(RRF, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=0)
randomcv_rf = randomcv_rf.fit(X_train,y_train)
predictions_RRF = randomcv_rf.predict(X_test)

view_best_params_RRF = randomcv_rf.best_params_

best_model = randomcv_rf.best_estimator_

predictions_RRF = best_model.predict(X_test)

#R^2
RRF_r2 = r2_score(y_test, predictions_RRF)
#Calculating RMSE
RRF_rmse = np.sqrt(mean_squared_error(y_test,predictions_RRF))

print('Random Search CV Random Forest Regressor Model Performance:')
print('Best Parameters = ',view_best_params_RRF)
print('R-squared = {:0.2}.'.format(RRF_r2))
print('RMSE = ',RRF_rmse)


# In[94]:


gb = GradientBoostingRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(gb.get_params())


# In[95]:


##Random Search CV on gradient boosting model

gb = GradientBoostingRegressor(random_state = 0)
n_estimator = list(range(1,20,2))
depth = list(range(1,100,2))

# Create the random grid
rand_grid = {'n_estimators': n_estimator,
               'max_depth': depth}

randomcv_gb = RandomizedSearchCV(gb, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=0)
randomcv_gb = randomcv_gb.fit(X_train,y_train)
predictions_gb = randomcv_gb.predict(X_test)

view_best_params_gb = randomcv_gb.best_params_

best_model = randomcv_gb.best_estimator_

predictions_gb = best_model.predict(X_test)

#R^2
gb_r2 = r2_score(y_test, predictions_gb)
#Calculating RMSE
gb_rmse = np.sqrt(mean_squared_error(y_test,predictions_gb))

print('Random Search CV Gradient Boosting Model Performance:')
print('Best Parameters = ',view_best_params_gb)
print('R-squared = {:0.2}.'.format(gb_r2))
print('RMSE = ', gb_rmse)


# In[96]:


from sklearn.model_selection import GridSearchCV    
## Grid Search CV for random Forest model
regr = RandomForestRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_rf = GridSearchCV(regr, param_grid = grid_search, cv = 5)
gridcv_rf = gridcv_rf.fit(X_train,y_train)
view_best_params_GRF = gridcv_rf.best_params_

#Apply model on test data
predictions_GRF = gridcv_rf.predict(X_test)

#R^2
GRF_r2 = r2_score(y_test, predictions_GRF)
#Calculating RMSE
GRF_rmse = np.sqrt(mean_squared_error(y_test,predictions_GRF))

print('Grid Search CV Random Forest Regressor Model Performance:')
print('Best Parameters = ',view_best_params_GRF)
print('R-squared = {:0.2}.'.format(GRF_r2))
print('RMSE = ',(GRF_rmse))


# In[97]:


## Grid Search CV for gradinet boosting
gb = GradientBoostingRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_gb = GridSearchCV(gb, param_grid = grid_search, cv = 5)
gridcv_gb = gridcv_gb.fit(X_train,y_train)
view_best_params_Ggb = gridcv_gb.best_params_

#Apply model on test data
predictions_Ggb = gridcv_gb.predict(X_test)

#R^2
Ggb_r2 = r2_score(y_test, predictions_Ggb)
#Calculating RMSE
Ggb_rmse = np.sqrt(mean_squared_error(y_test,predictions_Ggb))

print('Grid Search CV Gradient Boosting regression Model Performance:')
print('Best Parameters = ',view_best_params_Ggb)
print('R-squared = {:0.2}.'.format(Ggb_r2))
print('RMSE = ',(Ggb_rmse))


# # Prediction of fare from provided test dataset :
# We have already cleaned and processed our test dataset along with our training dataset. Hence we will be predicting using grid search CV for random forest model

# In[98]:



## Grid Search CV for random Forest model
regr = RandomForestRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator,
             'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_rf = GridSearchCV(regr, param_grid = grid_search, cv = 5)
gridcv_rf = gridcv_rf.fit(X_train,y_train)
view_best_params_GRF = gridcv_rf.best_params_

#Apply model on test data
predictions_GRF_test_Df = gridcv_rf.predict(test)


# In[99]:


predictions_GRF_test_Df


# In[100]:


test['Predicted_fare'] = predictions_GRF_test_Df


# In[101]:


test.to_csv('test.csv')


# In[103]:


test.head()


# In[ ]:




