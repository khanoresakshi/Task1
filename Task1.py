#!/usr/bin/env python
# coding: utf-8

# # Task 1: Predict the percentage of student based on no. of study hours

# In[4]:


# importing the libraries
import pandas as pd  
import numpy as np    
import matplotlib.pyplot as plt 


# In[5]:


# importing the data
url="http://bit.ly/w-data"
data=pd.read_csv(url)
data.head(5)


# Now we will check if there is any relationship between the variables of the dataset. We can create the plot for the same.

# In[21]:


data.plot(x='Hours', y='Scores',style='o')  
plt.title('Score vs Hours')
plt.xlabel('No. of hours')
plt.ylabel('Percentage Scored')
plt.show()
print(data.corr())


# From the above graph, we can clearly see that there is a positive linear relation between the variables.

# Training the model
# 1) Splitting the data

# In[30]:


# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# 2) Fitting the data into the model

# In[31]:


regression = LinearRegression()
regression.fit(train_X, train_y)
print("---------Model Trained---------")


# In[23]:


# Plotting the regression line
line = regressor.coef_*x+regressor.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line);
plt.show()
print(data.corr())


# Predicting the percentage of marks

# In[32]:


pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# In[33]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[35]:


hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))


# In[ ]:




