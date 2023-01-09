#%%
##INTRODUCTION
'''The objective of the study is to analyze the flight booking dataset obtained from 
“Ease My Trip” website and to conduct various statistical hypothesis tests in 
order to get meaningful information from it. The 'Linear Regression' statistical 
algorithm would be used to train the dataset and predict a continuous target variable. 
'Easemytrip' is an internet platform for booking flight tickets, and hence a platform 
that potential passengers use to buy tickets. A thorough study of the data will aid in 
the discovery of valuable insights that will be of enormous value to passengers.'''

##Research Questions
'''The aim of our study is to answer the below research questions:
a) Does price vary with Airlines?
b) How is the price affected when tickets are bought in just 1 or 2 days before departure?
c) Does ticket price change based on the departure time and arrival time?
d) How the price changes with change in Source and Destination?
e) How does the ticket price vary between Economy and Business class?'''

##References = https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

#%%
import time
start = time.time()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score,  mean_squared_error
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder

from scipy.stats import bartlett
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR

from logs_code import *
import pickle
# %%
logger = log(path="C:/Users/subodhkumar.n/Desktop/ML Practice/Flight-Price-Prediction", file="flights_pred_logs.log")

#%%
# Reading what is in the Data
df=pd.read_csv('Clean_Dataset.csv')
df.head()
#%%
# It is clear that Unnamed columns are not useful
# so dropping the useless column 'Unnamed: 0'
df=df.drop('Unnamed: 0',axis=1)
# %%
# A Quick Information about the Data
df.info()
# %%
#In this dataset we have around 300k rows. In order to reduce the runtime I want to take a sample portion
#There are different sampling methods like, random sampling, conditional sampling, sampling at constant rate.
#Here in this case I want to use a random sampling method.
#I want to take 50% data points from this population dataset. We can take sample by percentage or number(no. of rows).
sub_df = df.sample(frac=0.5)
sub_df_backup = sub_df.copy()
# %%
# Statistical Description of Data
sub_df.describe()
# %%
##Observations
#So now we have 150076 samples after taking 50% of population
#For both 'duration' and 'days_left' mean and median values are almost the same.
#But for 'price' there is big difference in  mean and median(50%) values.
plt.figure(figsize = (10,20))
plt.subplot(2,1,2)
sns.histplot(x = 'price', data = sub_df, kde = True)
plt.subplot(2,2,1)
sns.boxplot(x = 'price', data = sub_df)
# %%
#Even though the mean is around 20000, we can see here that the median 
#is approximately 7500. This difference is explainable by the presence 
#of two different tickets: business and economy. On the second graph, we 
#can see that the dispersion seems to be composed by two gaussian curves.

##How does the ticket price vary between Economy and Business class?

plt.figure(figsize=(20, 15))
sns.barplot(x='airline',y='price',hue="class",data=sub_df.sort_values("price")).set_title('Airline prices based on the class and company')
plt.legend(loc='upper left', frameon=True, fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel('Airline', fontsize=16);
plt.ylabel('Price', fontsize=16);
plt.show()
# %%
#Business flights are only available in two companies: 
# Air India and Vistara. Also, there is a big gap between the 
# prices in the two class that reaches almost 5 times the price 
# of Economy for Business tickets.

##Checking for airlines that has more flights
sub_df1=sub_df.groupby(['flight','airline'],as_index=False).count()
sub_df1.airline.value_counts()
# %%
# Indigo is the most popular airlines
sub_df2=sub_df.groupby(['flight','airline','class'],as_index=False).count()
sub_df2['class'].value_counts()
# %%
#Most of the Airlines has Economic Class as common.

##Does price vary with Airlines?
plt.figure(figsize=(15,10))
sns.boxplot(x=sub_df['airline'],y=sub_df['price'],palette='hls')
plt.title('Airlines Vs Price',fontsize=15)
plt.xlabel('Airline',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.show()
# %%
#There are slight differences between each companies on this graph, AirAsia 
# seems to have the cheapest flights when Air India and Vistara are more expensive.

##How Does the Ticket Price vary between Economy and Business Class?
plt.figure(figsize=(10,5))
sns.boxplot(x='class',y='price',data=sub_df,palette='hls')
plt.title('Class Vs Ticket Price',fontsize=15)
plt.xlabel('Class',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.show()
# %%
#There is a huge variation in the price of economy and business class tickets.
sns.violinplot(y = "price", x = "airline",data = sub_df.loc[df["class"]=='Business'].sort_values("price", ascending = False), kind="boxen")
plt.title("Airline prices based on companies for business tickets",fontsize=20)
# %%
#It looks like Vistara's business tickets are a little more expensive than the Air India's ones.
#How does the Ticket Price vary with the number of stops of a Flight?
plt.figure(figsize=(10,5))
sns.boxplot(x='stops',y='price',data=sub_df,palette='hls')
plt.title('Stops Vs Ticket Price',fontsize=15)
plt.xlabel('Stops',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.show()
# %%
sns.barplot(y = "price", x = "airline",hue="stops",data = sub_df.loc[df["class"]=='Economy'].sort_values("price", ascending = False))
plt.title("Airline prices based on the number of stops  for economy",fontsize=20)
# %%
sns.barplot(y = "price", x = "airline",hue="stops",data = sub_df.loc[df["class"]=='Business'].sort_values("price", ascending = False))
plt.title("Airline prices based on the number of stops  for business",fontsize=20)
# %%
#It's clear that the more stops there are the more expensive the flight is 
#except for AirAsia where the prices seems more constant. The behaviour and 
#different analysis of AirAsia tend to show that it relates to a low cost company.

##How the Ticket Price change based on the Departure Time and Arrival Time?
#Checking based on the Departure Time.
plt.figure(figsize=(25,15))
sns.boxplot(x='departure_time',y='price',data=sub_df)
plt.title('Departure Time Vs Ticket Price',fontsize=20)
plt.xlabel('Departure Time',fontsize=20)
plt.ylabel('Price',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.show()
# %%
#Checking based on the arrival time.
plt.figure(figsize=(25,10))
sns.boxplot(x='arrival_time',y='price',data=sub_df,palette='hls')
plt.title('Arrival Time Vs Ticket Price',fontsize=20)
plt.xlabel('Arrival Time',fontsize=20)
plt.ylabel('Price',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.show()
# %%
##Observations
#Departure Time Vs Ticket Price
#Ticket Price is More for the Flights when the Departure Time is at Night
#Ticket Price is almost equal for flights Having Departure time at Early_morning , Morning and Evening
#Ticket Price is Low for the Flights Having Departure Time at Late_night

# Arrival Time Vs Ticket Price
#Ticket Price is More for the Flights when the Arrival Time is at Evening
#Ticket Price is almost equal for flights Having Arrival time is at Morning and Night
#Ticket Price is Low for the Flights Having Arrival Time at Late_night as same as Departure Time
# %%
#How the price changes with change in Source city and Destination city?
ax = sns.relplot(col="source_city", y="price", kind="line",x='destination_city', data=sub_df, col_wrap=3)
ax.fig.subplots_adjust(top=1) # adjust the Figure in rp
ax.fig.suptitle('Airline prices based on the source and destination cities',fontsize=20)
# %%
#On one hand, it seems that flight leaving from Delhi are often 
# cheaper that from other source cities and the capital is also the 
# cheapest destination to go probably because as a capital cities, 
# the airport is the biggest and proposes more flights. In an other 
# hand, the prices are more or less similar and Hyderabad being the 
# most expensive destination.
# %%
#How does the price affected on the days left for Departure?
plt.figure(figsize=(25,10))
sns.lineplot(data=sub_df,x='days_left',y='price',color='blue')
plt.title('Days Left For Departure Versus Ticket Price',fontsize=20)
plt.xlabel('Days Left for Departure',fontsize=20)
plt.ylabel('Price',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.show()
# %%
plt.figure(figsize=(20,10))
sns.lineplot(data=sub_df,x='days_left',y='price',color='blue',hue='airline',palette='hls')
plt.title('Days Left For Departure Versus Ticket Price of each Airline',fontsize=15)
plt.legend(fontsize=15)
plt.xlabel('Days Left for Departure',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.tick_params(axis='both', labelsize=18)
plt.show()
# %%
#Observation
#As we can see when compared to others when there are two days 
# remaining for departure then the Ticket Price is very High 
# for all the airlines. The prices of Vistara and Air india fall before 1 day.
# From day 20 prices of all airlines are increasing rapidly.  
# %%
#Total no. of flights from one city to another
sub_df.groupby(['flight','source_city','destination_city','airline','class'],as_index=False).count().groupby(['source_city','destination_city'],as_index=False)['flight'].count()
# %%
#Average Price of different Airlnes from Source city to Destination city
df.groupby(['airline','source_city','destination_city'],as_index=False)['price'].mean().head(10)
# %%
#Does the price change with the duration of the flight?
df_temp = sub_df.groupby(['duration'])['price'].mean().reset_index()

plt.figure(figsize=(15,6))
ax = sns.scatterplot(x="duration", y="price", data=df_temp).set_title("Average prizes depending on the duration",fontsize=15)
# %%
ax = sns.scatterplot(x="duration", y="price", data=df_temp).set_title("Average prizes depending on the duration",fontsize=15)
ax = sns.regplot(x="duration", y="price", data=df_temp, order=2)
# %%
##Observation
#It is clear that here the relationship is not linear but can be approximated 
# with a second degree curve. The prices reaches a high price at a duration 
# of 20 hours before lowering again. However some outliers seem to affect 
# the regression curve 
# %%
sub_df.info()
# %%
#It is clear that columns datatype are objects. While most of the Ml algorithms
# only work with numeric values. Since these categorical features cannot be directly 
# used in most machine learning algorithms, the categorical features need to 
# be transformed into numerical features.While numerous techniques exist to 
# transform these features, the most common technique is one-hot encoding.
# In one-hot encoding, a categorical variable is converted into a set of 
# binary indicators (one per category in the entire dataset).
# %%
sub_df_outlier = sub_df.copy()
index_price = sub_df[sub_df['price'] >= 90000].index
sub_df_outlier.drop(index_price, inplace=True)
sub_df_outlier.head()
sub_df_outlier.info()
# %%
sns.boxplot(x = 'price', data = sub_df_outlier)

# what looks like an outliers for the ticket price is an valuable data and we 
#can't simply delect it. Here we are comparing of 2 classes which has a very big
# variation in terms of price which is close to 5 times. 
# And we have only 295 of business class and if try remove anything the quality 
# of model will affected in terms of predicting the business class tickets.
# And this prices are also affected by the days_left variable.
# %%
# Converting the labels into a numeric form using Label Encoder
def preprocessing(df):
    #Encode the ordinal variables "stops" and "class".
    df["stops"] = df["stops"].replace({'zero':0,'one':1,'two_or_more':2}).astype(int)
    df["class"] = df["class"].replace({'Economy':0,'Business':1}).astype(int)
    
    #Create the dummy variables for the cities, the times and the airlines.
    dummies_variables = ["airline","source_city","destination_city","departure_time","arrival_time"]
    dummies = pd.get_dummies(df[dummies_variables], drop_first= True)
    df = pd.concat([df,dummies],axis=1)
    
    #Create the dummy variables for the cities, the times and the airlines.
    df = df.drop(["flight","airline","source_city","destination_city","departure_time","arrival_time"],axis=1)
    
    return df
# %%
df_preprocessed = preprocessing(sub_df)
# %%
df_preprocessed.head()
# %%
#Checking for the correlation
plt.figure(figsize=(50,50))
res = sns.heatmap(df_preprocessed.corr(),annot = True, annot_kws={'size': 20}, vmin= -1.0, vmax= 1.0, center = 0, cmap = 'RdBu_r')
res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 25, rotation =90)
res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 25, rotation =360)
plt.show()
# %%
#However, the correlation is a good metric for linear relationship, but 
# doesn't highlight non linear ones. For that I will use mutual information.
# %%

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        # _ Ignore a value of specific location/index
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
#first element of sol series is the pair with the biggest correlation
# %%
X = df_preprocessed.copy()
y = X.pop("price")
# %%
mi_scores = make_mi_scores(X, y)
print(mi_scores.sort_values(ascending=False))
# %%
##Observation
#This importance analysis shows us the class or the fast that the flight 
# is a Vistara one but vairables like the duration of the flight or the 
# number of days left have strong non linear relationship too as we saw 
# on the previous questions.
# %%
m, (f1, f2) = plt.subplots(1, 2, sharey=True, figsize = (20,5) ) #ploting price with year, days_left, mileage
f1.scatter(df_preprocessed['duration'], df_preprocessed['price'])
f1.set_title('price Vs duration')
f2.scatter(df_preprocessed['days_left'], df_preprocessed['price'])
f2.set_title('price Vs days_left')
plt.show()
# %%
#Checking for the multicolinearity, as per the linear regression assumptions theres 
#should be no colinearity among the independent variables(x1, x2, x3..)
from statsmodels.stats.outliers_influence import variance_inflation_factor #importing the library to calculate VIF
variables = X #defining the features to check
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
# %%
vif
# %%
#As no VIF is greater than 10, which means all variables are independent to each other.

# %%
logger.info("Start Model Building")
##Prediction of the flight prices
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =1, shuffle = True)

# %%
#Although normalization via min-max scaling is a commonly used technique that 
# is useful when we need values in a bounded interval, standardization can be 
# more practical for many machine learning algorithms. The reason is that many 
# linear models, such as the logistic regression and SVM, [...] initialize 
# the weights to 0 or small random values close to 0. Using standardization, 
# we center the feature columns at mean 0 with standard deviation 1 so that 
# the feature columns take the form of a normal distribution, which makes it 
# easier to learn the weights. Furthermore, standardization maintains useful 
# information about outliers and makes the algorithm less sensitive to them 
# in contrast to min-max scaling, which scales the data to a limited range 
# of values.
# %%
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.fit_transform(x_test)
x_train_scaled=pd.DataFrame(x_train_scaled, columns = x_train.columns)
x_test_scaled=pd.DataFrame(x_test_scaled, columns = x_test.columns)
# %%
x_train_scaled.head()
# %%
lr_start = time.time()
logger.info("Start of Linear Regression Model")
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)
y_pred = lr.predict(x_test_scaled)

#Checking for the overfitting and the underfitting issues
logger.info("Training Accuracy {} ".format(r2_score(y_train,lr.predict(x_train_scaled))*100))
logger.info("Validation Accuracy {}".format(r2_score(y_test,lr.predict(x_test_scaled))*100))



#High training accuracy, low test accuracy = Overfitting(high variances, low bias)
#Low training accuracy, high test accuracy = Underfitting(high bias, low variances)
#Bias: A difference occurs between prediction values made by the model and actual 
# values/expected values, and this difference is known as bias errors or 
# Errors due to bias.
#Variances: variance tells that how much a random variable is different from 
# its expected value.
# %%
mae = mean_absolute_error(y_test, y_pred)
print('MAE score:', mae)
logger.info('MAE of LR Model {}'.format(mae))

#MAE is a very simple metric which calculates the absolute difference between 
# actual and predicted values. The MAE of your model which is basically a 
# mistake made by the model known as an error. So, sum all the errors and 
# divide them by a total number of observations And this is MAE. And we aim 
# to get a minimum MAE because this is a loss.

r2 = r2_score(y_test, y_pred)
print('R2 score:', r2)
logger.info('R2 Score score of LR Model {}'.format(r2))

#The disadvantage of the R2 score is while adding new features in data the 
# R2 score starts increasing or remains constant but it never decreases 
# because It assumes that while adding more data variance of data increases.
#r2 = 1-(sum of squared error of regression line/squared sum error of mean line)

#rmse = mean_squared_error(y_test, y_pred, squared=False)
#Setting squared to False will return the RMSE.
#print('RMSE score:', rmse)

# adjusted R-squared
adj_r2 = 1 - (1-r2_score(y_test, y_pred)) * (len(y)-1)/(len(y)-X.shape[1]-1)
print('adjusted_R-squared:', adj_r2)
logger.info("Adjusted R2 score of LR Model {}".format(adj_r2))

#The disadvantage of the R2 score is while adding new features in data the R2 
# score starts increasing or remains constant but it never decreases because 
# It assumes that while adding more data variance of data increases.
#But the problem is when we add an irrelevant feature in the dataset then at 
# that time R2 sometimes starts increasing which is incorrect.

lr_end = time.time()
print('Linear Regression time: ', (lr_end - lr_start))

# %%
residuals = y_test.values-y_pred
mean_residuals = np.mean(residuals)
print("Mean of Residuals:", mean_residuals)
logger.info('Mean Residuals of LR Model {}'.format(mean_residuals))
#Residuals as we know are the differences between the true value and the 
# predicted value. One of the assumptions of linear regression is that the 
# mean of the residuals should be zero
# %%
#Check for Homoscedasticity
#Homoscedasticity means that the residuals have equal or almost equal 
# variance across the regression line. By plotting the error terms with 
# predicted terms we can check that there should not be any pattern in the 
# error terms.
plt.scatter(y_pred, residuals)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.plot(y_pred, [0]*len(y_pred), color='red')
hmsc_test = bartlett( y_pred,residuals)
print(hmsc_test)

logger.info("Linear Regression Model is done.")
# %%
#It is clear that error is not constant across the values of the dependent variables.

#checking with another alogrithm

logger.info("Start of Random Forest Regression")

rfr_start = time.time()
rfr = RandomForestRegressor()
rfr.fit(x_train_scaled, y_train)
y_pred_rfr = rfr.predict(x_test_scaled)

#Checking for the overfitting and the underfitting issues
logger.info('Training Accuracy {} '.format(r2_score(y_train,rfr.predict(x_train_scaled))*100))
logger.info('Validation Accuracy {} '.format(r2_score(y_test,rfr.predict(x_test_scaled))*100))
# %%
mae_rfr = mean_absolute_error(y_test, y_pred_rfr)
print('MAE score:', mae_rfr)
logger.info('MAE of RFR Model{}'.format(mae_rfr))

r2_rfr = r2_score(y_test, y_pred_rfr)
print('R2 score:', r2_rfr)
logger.info('R2 score of RFR Model {}'.format(r2_rfr))

adj_r2_rfr = 1 - (1-r2_score(y_test, y_pred_rfr)) * (len(y)-1)/(len(y)-X.shape[1]-1)
print('adjusted_R-squared:', adj_r2)
logger.info('Adjusted R2 score of RFR Model {}'.format(adj_r2_rfr))

rfr_end = time.time()
print('Random Forest Regressot time:', (rfr_end - rfr_start))

logger.info("End of the Random Forest Regression Model")
# %%

logger.info("Start of Stacking Model")
#Checking with stacking
stack_start = time.time()

estimators = [('lr', LinearRegression()), ('etr', ExtraTreesRegressor())]

stack_reg = StackingRegressor(estimators=estimators,final_estimator=RandomForestRegressor(n_estimators=10,random_state=1))

stack_reg.fit(x_train_scaled, y_train)

# %%
y_pred_stack = stack_reg.predict(x_test_scaled)
#Checking for the overfitting and the underfitting issues
logger.info('Training Accuracy {} '.format(r2_score(y_train,stack_reg.predict(x_train_scaled))*100))
logger.info('Validation Accuracy {} '.format(r2_score(y_test,stack_reg.predict(x_test_scaled))*100))
# %%

mae_stack = mean_absolute_error(y_test, y_pred_stack)
print('MAE score:', mae_stack)
logger.info('MAE of Stacking Model {}'.format(mae_stack))

r2_stack = r2_score(y_test, y_pred_stack)
print('R2 score:', r2_stack)
logger.info('R2 score of Stacking Model {}'.format(r2_stack))

adj_r2_stack = 1 - (1-r2_score(y_test, y_pred_stack)) * (len(y)-1)/(len(y)-X.shape[1]-1)
print('adjusted_R-squared:', adj_r2_stack)
logger.info('Adjusted R2 score of Stacking Model {}'.format(adj_r2_stack))

stack_end = time.time()
print('Stacking Model time:', (stack_end - stack_start))
# %%
## Take away points:
#1. The model that gives the best result is the Random Forest Regressor with
# on the test dataset an R^2 score equals to 0.9808 and a MSE score equals 
# to 1398.37.

#2. There is a big gap between flight tickets in business and economy. In 
# average business tickets are 6.5 times more expensive than economy tickets.

#3. Vistara and AirIndia seems to be the most expensive companies and 
# AirAsia the cheapest. However for business tickets, only Vistara and 
# AirIndia are available, and Vistara is slightly more expensive.

#4. In general, prices rise quite slowly until 20 days before the flight 
# where the prices rise drastically. But one day before the flight, 
# there usually are empty seats that have not been sold. Thus it is possible 
# to find tickets three times cheaper than the day before.

#5. The longer the flight is the more expensive the tickets are until it 
# reaches around 20 hours, then the prices tend to decrease.

#6. For the time of the flight:
# It seems that departure during the afternoon and late night are cheaper, 
# and night more expensive.
#It seems that departure during the early morning, afternoon and late night 
# are cheaper, and evening more expensive.

# 7. For the cities of the trip:
# Flights from Delhi are the cheapest the from the others cities seems equal 
# on average but slightly more expensive for Chenai.
# Flight to Delhi are the cheapest and to Bengalore the most expensive ones.

# 8. In general, the more stops there are, the more expensive the 
# flight ticket is.

end = time.time()
print('Total Time: ', (end - start))

#saving the model to resue.

# %%
