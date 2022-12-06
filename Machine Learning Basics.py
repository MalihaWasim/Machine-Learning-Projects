#!/usr/bin/env python
# coding: utf-8

# In[1]:


# A data set
# An example of an array
[99,85,88,77,152,45,95,83,79,103]


# In[2]:


# Types of Data
#numerical, categorical and ordinal


# In[3]:


# numerical data (discrete and continous)


# In[4]:


# categorical data (yes/no values, color values)


# In[5]:


# ordinal data is like a categorical data but in ordered form


# ### Mean Median Mode

# In[7]:


speed = [99,85,88,77,152,45,95,83,79,103]


# In[11]:


# Use the Numpy mean() method to take the average speed

import numpy

speed = [99,85,88,77,152,45,95,83,79,103]

x = numpy.mean(speed)

print(x)


# In[13]:


# Use the Numpy median() method to find the middle value (for odd)

import numpy
speed = [99,85,88,77,152,45,95,83,79]
x = numpy.median(speed)
print(x)


# In[14]:


# Use the Numpy median() method to find the middle value (for even)

import numpy
speed = [99,85,88,77,152,45,95,83]
x = numpy.median(speed)
print(x)


# In[18]:


# for mode, use the Scipy mode() method to get the number that appears the most

from scipy import stats
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
x = stats.mode(speed)
print(x)


# ### Standard Deviation

# In[21]:


# we want to check how spread out the values in our data set are.
# low S.D means that most of the numbers are close to the mean value
# high means thta the values are spread over a wider range


# In[22]:


# this time we have registered the speed of cars
speed = [86,97,84,85,91]


# In[23]:


import numpy
speed = [86,97,84,85,91]
x = numpy.std(speed)
print(x)


# In[24]:


import numpy
price = [500,899,1000]
x = numpy.std(price)
print(x)


# In[25]:


# variance means taking the sq root of standard deviation
import numpy
speed = [86,97,84,85,91]
x = numpy.var(speed)
print(x)


# ### Percentiles

# In[31]:


# numpy percentile() method to find the percentiles
# lets say how many senior citizens are living in our street
# what is the 75. percentile?
import numpy
ages = [23,78,45,90,56,75,56,3,28,60,35,86,50]
x = numpy.percentile(ages, 75)
print(x)
# ans is 75, meaning that 75% of the people are senior citizens who live in the street


# In[36]:


# what is the age that 90% of the people are younger than?
import numpy
ages = [5,32,45,49,51,17,15,12,39,6,25,36,13,48]
x = numpy.percentile(ages, 90)
print(x)


# ## Data Distribution 
# ### used to create big data randomly

# In[38]:


# create an array containing 255 random floats between 0 and 5
# creating a random data set of 255 random floats between 0 and 5
import numpy
x = numpy.random.uniform(0.0, 5.0, 255)
print(x)


# ### Histogram

# In[42]:


# matplot lib is used 
# draw a histogram

import numpy
import matplotlib.pyplot as plt

x = numpy.random.uniform(0.0, 5.0, 255)

plt.hist(x, 5)
plt.show()


# ### Big Data Distributions

# In[45]:


# create an array with 1000000 random numbers, and display them using a histogram with 1000 bars
import numpy 
import matplotlib.pyplot as plt

x = numpy.random.uniform(0.0, 5.0, 1000000)
plt.hist(x, 1000)
plt.show()


# In[ ]:




