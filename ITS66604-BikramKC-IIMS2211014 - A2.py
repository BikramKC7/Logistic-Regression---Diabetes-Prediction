#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


# In[2]:


df = pd.read_csv('diabetes.csv')
df


# In[3]:


df.info()


# In[4]:


df['BloodPressure'].replace(0,np.nan,inplace = True)
df['SkinThickness'].replace(0,np.nan,inplace = True)
df['Insulin'].replace(0,np.nan,inplace = True)
df['BMI'].replace(0,np.nan,inplace = True)
df


# In[5]:


df.isnull().sum()


# In[6]:


mean_BloodPressure = df['BloodPressure'].astype('float').mean(axis = 0)
df['BloodPressure'].replace(np.nan, mean_BloodPressure, inplace=True)

mean_SkinThickness = df['SkinThickness'].astype('float').mean(axis=0)
df['SkinThickness'].replace(np.nan, mean_SkinThickness, inplace=True)

mean_Insulin = df['Insulin'].astype('float').mean(axis=0)
df['Insulin'].replace(np.nan, mean_Insulin, inplace=True)

mean_BMI = df['BMI'].astype('float').mean(axis=0)
df['BMI'].replace(np.nan, mean_BMI, inplace=True)

print(mean_BloodPressure)
print(mean_SkinThickness)
print(mean_Insulin)
print(mean_BMI)


# In[7]:


df[['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].astype('float')
df


# In[8]:


df.info()


# In[9]:


corr_matrix = df.corr()
corr_matrix


# In[10]:


corr_values = corr_matrix["Outcome"].abs()
corr_values


# In[11]:


sorted_corr = corr_values.sort_values(ascending= False)
sorted_corr


# In[12]:


X = df[["Glucose"]]
y = df["Outcome"]


# In[13]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size =0.20, random_state = 34) 


# In[15]:


model = LogisticRegression()
model.fit(X_train,y_train)


# In[16]:


y_pred = model.predict(X_test)
y_pred


# In[17]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:




