#!/usr/bin/env python
# coding: utf-8

# GRIP : Data Science and Business Analytics Intern at The Spark Foundation 

# # Pridiction using the Supervised Learning 

# # Author : Anurag Chaurasia

# # Objective
# 
# Predict the percentage of an student based on the number of study hours.
# 
# 

# Importing all the required libraries

# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Dataset URL

# In[3]:


data="http://bit.ly/w-data"


# In[11]:


df=pd.read_csv(data)


# In[12]:


df.info


# In[13]:


df.head()


# Plotting the Data on Scatter Plot

# In[22]:


df.plot(x="Hours",y="Scores",style="o",C='red')
plt.title("Hours vs Scores",size=18)
plt.xlabel("No. of Hours",size=15)
plt.ylabel("Scores obtained(%)",size=15)
plt.show()


# # Preparing the Data for the Pridiction

# In[35]:


#deviding the data into attributes and labels 
X = df.iloc[:, :-1].values  
Y = df.iloc[:, 1].values


# In[33]:


print(X) 


# In[37]:


print(Y)


# Splitting the dataset into training data and the testing data using Scikit-Learn's

# In[38]:


from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=0) 


# # Training Algorithm

# In[45]:


#Importing the library for the model 
from sklearn import linear_model

#Creating the linear regration object
linear=linear_model.LinearRegression()

#Trainig the model using the trainig set and check score
linear.fit(X,Y)
linear.score(X,Y)


# # Equation for Coefficient and Intercept

# In[48]:


print("Cofficient: ",linear.coef_)
print("Intercept: ",linear.intercept_)


# # Plotting the Model

# In[99]:


# plotting the Regrassion Line
line = linear.coef_*X+linear.intercept_

# Plotting  the test dataset
plt.scatter(X,Y,c='red',label="Scores");
plt.plot(X, line,c='blue');
plt.legend()


# # Prediction using the Model

# In[96]:


print(X_test)
Y_prediction = linear.predict(X_test)


# In[79]:


# Comparing the Actual data and the Predicted data
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_prediction})  
df 


# # Testing the Model

# In[92]:


hours = [[9.25]]
pred = linear.predict(hours)

print("No. of Hours/day =",hours[0])
print("Predicted Scores = ",pred[0])


# # Evaluating the Model

# In[97]:


from sklearn import metrics
from sklearn.metrics import r2_score
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_prediction))
print('R2 Score:',r2_score(Y_test,Y_prediction))


# In[ ]:




