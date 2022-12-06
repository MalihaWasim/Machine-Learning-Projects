#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Problem Statement 
# we have a logistic regression object that is ready to check whether a tumor us cancerous based on the tumor size


# In[4]:


import numpy
from sklearn import linear_model

# reshaped for logistic function
X = numpy.array([3.78, 2.44, 2.09, 0.17, 1.45, 1.62, 4.87, 4.96]).reshape(-1,1)
y = numpy.array([0,0,0,0,1,1,1,1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)

# predict if tumor is cancerous where the size is 3.46mm
predicted = logr.predict(numpy.array([3.46]).reshape(-1,1))
print(predicted)


# In[5]:


# result shows that we have predicted that a tumor with a size of 3.46mm will be cancerous


# In[6]:


import numpy
from sklearn import linear_model

# reshaped for logistic function
X = numpy.array([3.78, 2.44, 2.09, 0.17, 1.45, 1.62, 4.87, 4.96]).reshape(-1,1)
y = numpy.array([0,0,0,0,1,1,1,1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)

log_odds = logr.coef_
odds = numpy.exp(log_odds)

print(odds)


# In[7]:


# This tells us that as the size of a tumor increases 
# by 1mm the odds of it being a tumor increases by 1.4x.


# In[8]:


# Let us now use the function with what we have learned to find out the probability that each tumor is cancerous
import numpy
from sklearn import linear_model

# reshaped for logistic function
X = numpy.array([3.78, 2.44, 2.09, 0.17, 1.45, 1.62, 4.87, 4.96]).reshape(-1,1)
y = numpy.array([0,0,0,0,1,1,1,1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)

def logit2prob(logr, X):
    log_odds = logr.coef_ * X + logr.intercept_
    odds = numpy.exp(log_odds)
    probability = odds / (1+odds)
    return(probability)

print(logit2prob(logr, X))


# In[ ]:


# results
# 3.78 0.60 the probability that a tumor with the size 3.78cm is cancerous is 60%
# 2.44 0.48 the probability that a tumor with the size 2.44cm is cancerous is 48%
# 2.09 0.45 the probability that a tumor with the size 2.09cm is cancerous is 45%

