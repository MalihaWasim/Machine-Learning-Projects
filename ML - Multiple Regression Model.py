#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas

df = pandas.read_csv("CO2Emissiondata.csv")
print(df.to_string())


# In[2]:


X = df[['Weight', 'Volume']]
y = df['CO2']


# In[3]:


from sklearn import linear_model


# In[4]:


regr = linear_model.LinearRegression()
regr.fit(X, y)


# In[5]:


# predict the CO2 emission of a car where the weight is 1490kg, and the volume is 2000cm^3
predictedCO2 = regr.predict([[1490, 2000]])
print(predictedCO2)


# In[11]:


# so we have predicted that a car with 2.0 liter engine, and a weight of 1490 kg, 
# will release approximately 107 grams of CO2 for evry km it drives


# In[6]:


# Print the coefficient values of the regression object
import pandas
from sklearn import linear_model

df = pandas.read_csv("CO2Emissiondata.csv")
X = df[['Weight','Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)


# In[13]:


# the result values tell us if the weight increase by 1 kg, the CO2 emission increases by 0.00755095 g
# and if the engine size (Volume) increases by 1 cm3 engine weighs 1490kg, the CO2 emission will be approximately 107g


# In[7]:


# what if we increase the weight with 1000kg?
# copy the example before but change the weight from 1490 to 2490

import pandas
from sklearn import linear_model

df = pandas.read_csv("CO2Emissiondata.csv")
X = df[['Weight','Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict([[2490, 2000]])
print(predictedCO2)


# In[8]:


# results suggest that we have predicted that a car with 2.0 liter engine, and a weight of 2490 kg, 
# will release approximately 114 grams of CO2 for evry kilometer it drives
# which shows that the coefficient of 0.00755095 is correct
106.55614578 + (1000 * 0.00755095 )


# # Scale features

# ### these scaling features are used when the data has different values, and even different measurement units, 
# ### it can be difficult to compare them. like kg can not be compared to meters
# 

# In[10]:


# different methods are used for scaling
# here we are using standardization
# z = (x - u) / s
# z is the new value, x is the original value, u is the mean and s is the standard deviation 


# In[9]:


# if we take the weight column from the data above, the first value is 790, and the scaled value will be
# (790 - 1292.23) / 238.74 = -2.1
# Example - scale all values in the weight and volume coulumns
import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("CO2Emissiondata.csv")
X = df[['Weight', 'Volume']]
scaledX = scale.fit_transform(X)
print(scaledX)


# ### Predict CO2 Values

# In[19]:


# Now we will use the scaled data to predict values of CO2


# In[10]:


# Example - Predict the CO2 emission from a 1.3 liter car that weighs 2300 kilograms
import pandas 
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("CO2Emissiondata.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)


# # Train/Test

# In[23]:


# to measure if the predicted model is good enough, we can use a method called Train/Test
# it is used to measure the accuracy of the model
# the data is split into train set and testing set
# 80% for training and 20% for testing


# In[24]:


# train the model means creating the model
# test the model means test the accuracy of the model


# In[23]:


# taking a dataset which we want to test
# my data set illustrates 100 customers in a shop, and their shopping habits

import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

plt.scatter(x, y)
plt.show()


# In[26]:


# x axis represntes the no. of minutes before making a purchase
# y axis represnts the amount of money spent on the purchase


# ### split into Train/Test

# In[24]:


# the training set should be a random selection of 80% of the original data
# the testing set should be the remainging 20%
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]


# In[25]:


# display the training set
# in scatterplot with the training set
plt.scatter(train_x, train_y)
plt.show()


# In[26]:


# display the testing set
# to make sure the testing set is not entirely different, we will take a look at the testing set as well
plt.scatter(test_x, test_y)
plt.show()


# In[32]:


# testing set almost looks like the original data set 


# ### Fit the Data Set

# In[34]:


# if the model performs better on the training set (80%) than on the test set (20%),
# it means that the model is likely overfitting


# In[27]:


# the best fit in my opinion is the polynomial regression, 
# to draw a line through the data points, we use the plot() method of matplotlib module
# Example- Draw a polynomial regression line through the data points
import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

myline = numpy.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()


# In[15]:


# it seems like our data set is over fit as the line shows that the person who is spending 6 minutes in the shop would make a purchase worth 200


# In[28]:


# Now Using R-Squared Score as it is a good indicator of 
# how well the data set is fitting the model
# Example - How well does my training data fit in a polynomial regression
import numpy
from sklearn.metrics import r2_score
numpy.random.seed(2)
x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
r2 = r2_score(train_y, mymodel(train_x))

print(r2)


# In[29]:


# Now we are testing the model with the testing data to see if it gives the same result
# Find the R2 score when using testing data
import numpy
from sklearn.metrics import r2_score
numpy.random.seed(2)
x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
r2 = r2_score(test_y, mymodel(test_x))
print(r2)


# In[30]:


# we can Predict new values because our model is OK
# Example - How much money will a buying customer spend, if she or he stays in the shop for 5 minutes?
print(mymodel(5))


# In[33]:


# Results shows that the predicted value is 22.88 dollars which means that the customer is spending 22.88 dollars as seems to corresponds to the diagram


# In[ ]:




