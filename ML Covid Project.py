#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


# In[2]:


df = pd.read_csv('D:/OneDrive/Desktop/corona_tested_006.csv')


# In[3]:


df


# In[4]:


df.describe()


# In[6]:


df['Age_60_above']


# In[5]:


df.nunique()


# In[8]:


msno.matrix(df)


# In[9]:


unique_values = df['Age_60_above'].unique()


# In[10]:


unique_values


# In[12]:


df['Age_60_above'].isnull().sum()


# In[19]:


total_values = 278848
missing_values = 127320

percentage_missing = (missing_values / total_values) * 100
print(f"The percentage of missing values is {percentage_missing:.2f}%")


# In[13]:


sns.heatmap(df.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')


# In[23]:


mode_age_60_above = df['Age_60_above'].mode()[0]


# In[25]:


df['Age_60_above'].fillna(mode_age_60_above, inplace=True)


# In[26]:


sns.heatmap(df.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')


# In[28]:


df['Sex'].isnull().sum()


# In[29]:


mode_sex = df['Sex'].mode()[0]

df['Sex'].fillna(mode_sex, inplace=True)


# In[30]:


sns.heatmap(df.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')


# In[31]:


df


# In[34]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

binary_columns = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache']
for col in binary_columns:
    df[col] = label_encoder.fit_transform(df[col])


# In[35]:


df


# In[37]:


df['Corona'] = label_encoder.fit_transform(df['Corona'])
df['Sex'] = label_encoder.fit_transform(df['Sex'])


# In[38]:


df


# In[39]:


df['Age_60_above'] = label_encoder.fit_transform(df['Age_60_above'])


# In[40]:


df


# In[41]:


df['Known_contact'].unique()


# In[42]:


df = pd.get_dummies(df, columns=['Known_contact'])




# In[43]:


df


# In[44]:


df['Known_contact_Abroad'] = label_encoder.fit_transform(df['Known_contact_Abroad'])
df['Known_contact_Contact with confirmed'] = label_encoder.fit_transform(df['Known_contact_Contact with confirmed'])
df['Known_contact_Other'] = label_encoder.fit_transform(df['Known_contact_Other'])


# In[45]:


df


# In[46]:


df.info()


# In[51]:


sns.pairplot(df,hue = 'Corona')


# In[48]:





# In[54]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


features = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache', 'Age_60_above', 'Sex','Known_contact_Abroad','Known_contact_Contact with confirmed','Known_contact_Other']
target = 'Corona'

train_data, test_data, train_labels, test_labels = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(train_data, train_labels)

predictions = rf_classifier.predict(test_data)

accuracy = accuracy_score(test_labels, predictions)
conf_matrix = confusion_matrix(test_labels, predictions)
classification_rep = classification_report(test_labels, predictions)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')


# In[ ]:




