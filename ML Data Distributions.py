#!/usr/bin/env python
# coding: utf-8

# ### Normal Data Distribution

# In[3]:


import numpy 
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 100000)

plt.hist(x, 100)
plt.show()


# In[8]:


# mean = 7.0 and S.D = 3.0
import numpy
import matplotlib.pyplot as plt
x = numpy.random.normal(7.0, 3.0, 100000)
plt.hist(x, 100)
plt.show()


# ### Scatter Plot

# In[14]:


# need two arrays of same length
# x array represents age of each car
# y array represents the speed of each car
# use the scatter() method to draw a scatter plot diagram

import matplotlib.pyplot as plt
x = [5,7,9,11,6,2,4,8,4]
y = [75,86,84,102,78,95,94,87,85]

plt.scatter(x, y)
plt.show()


# ### Random Data Distributions

# In[17]:


# a scatter plot with 1000 dots
import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 1000)
y = numpy.random.normal(7.0, 3.0, 1000)

plt.scatter(x, y)
plt.show()


# In[ ]:




