#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import warnings

warnings.filterwarnings('ignore')


# In[2]:


#connecting with the file
con = sqlite3.connect(r"D:\data analysis\project\password_resources\password_data.sqlite")


# In[3]:


#reading the table
df = pd.read_sql_query('SELECT * FROM Users',con)
df.head(3)


# DATA CLEANING

# In[4]:


#Duplicated values
df.duplicated().sum()


# In[5]:


#missing values
df.isnull().sum()


# In[6]:


#removing irrelevant features
df.drop('index', axis=1, inplace = True)


# In[7]:


#removing irrelevant rows like rows with negative values

df_filtered = df[df['strength']>=0]
#df_filtered


# Data Analysis - Sematics Analysis

# In[38]:


#Password with only numeric
a = df['password'].str.isnumeric()                  
print("Password with only numeric-",df[a].shape[0])

#Password with only uppercase
b = df['password'].str.isupper()
print("Password with only uppercase-",df[b].shape[0])

#Password with only alphabet
c = df['password'].str.isalpha()
print("Password with only alphabet-",df[c].shape[0])

#Password with only alpha numeric
d = df['password'].str.isalnum()
print("Password with only alpha-numeric-",df[d].shape[0])

#Password with title case i.e first alphabet capital
e = df['password'].str.isnumeric()
print("Password with Title-Case-",df[e].shape[0])


# In[9]:


#password with special characters

import string

def pass_spec_char(row):
    for x in row:
        if x in string.punctuation:
            return 1
        else:
            pass

df[df['password'].apply(pass_spec_char)==1].shape


# Feature Engineering

# In[10]:


#password length
df['pass_len'] = df['password'].str.len()


# In[11]:


#frequency of upper, lower, digits and special characters

def freq_lowercase(password):
    return len([char for char in password if char.islower()])/len(password)

def freq_uppercase(password):
    return len([char for char in password if char.isupper()])/len(password)

def freq_numeric(password):
    return len([char for char in password if char.isnumeric()])/len(password)

def fre_spec_char(password):
    return len([char for char in password if not char.isnumeric() and not char.isalpha()])/len(password)


# In[12]:


df['freq_lowercase'] = df['password'].apply(freq_lowercase)
df['freq_uppercase'] = df['password'].apply(freq_uppercase)
df['freq_numeric'] = df['password'].apply(freq_numeric)
df['freq_spec_char'] = df['password'].apply(fre_spec_char)


# In[13]:


df.head(8)


# Analyzing the correlation between various feature and password and removing the useless feature

# In[14]:


column =['pass_len','freq_lowercase','freq_uppercase','freq_numeric','freq_spec_char'] 
    
for col in column:
    print(col)
    print(df[[col,'strength']].groupby('strength').aggregate(['max','min','mean','median'])) 
    print('\n')


# In[15]:


#for better visualization
df.columns


# In[16]:


fig,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize = (15,7))

sns.boxplot(y='pass_len', x ='strength',data = df, hue = 'strength', ax=ax1)
sns.boxplot(y='freq_lowercase', x ='strength',data = df, hue = 'strength', ax=ax2)
sns.boxplot(y ='freq_uppercase', x ='strength',data = df, hue = 'strength', ax=ax3)
sns.boxplot(y ='freq_numeric', x ='strength',data = df, hue = 'strength', ax=ax4)
sns.boxplot(y ='freq_spec_char', x ='strength',data = df, hue = 'strength', ax=ax5)


# In[17]:


#lets apply feature importance - univariet

def get_dist(df,features):
    plt.subplot(1,2,1)
    sns.violinplot(x = 'strength', y =features, data = df)
    
    plt.subplot(1,2,2)
    sns.distplot(x=df[df['strength']==0][features], color = 'red',label=0, hist = False)
    sns.distplot(x=df[df['strength']==1][features], color = 'orange',label=1, hist = False)
    sns.distplot(x=df[df['strength']==2][features], color = 'green',label=2, hist = False)
    plt.legend()
    plt.show()


# In[18]:


get_dist(df,'freq_lowercase')


# Apply TF-IDF on password 

# In[19]:


data = df.sample(frac=1)
data


# In[20]:


x = list(data['password'])

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='char')
X = vectorizer.fit_transform(x)


# In[21]:


X.shape


# In[22]:


vectorizer.get_feature_names_out() #columns on which the password is vectorized


# In[23]:


df_vector = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
df_vector


# Applying machine learning algo

# In[24]:


df_vector['pass_len'] = data['pass_len']
df_vector['freq_lowercase'] = data['freq_lowercase']


# In[25]:


df_vector


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X = df_vector
y = data['strength']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


# In[28]:


from sklearn.linear_model import LogisticRegression


# In[29]:


clf = LogisticRegression(multi_class='multinomial')


# In[30]:


clf.fit(X_train,y_train)


# In[31]:


y_pred = clf.predict(X_test)
y_pred


# In[32]:


from collections import Counter


# In[33]:


Counter(y_pred)


# Making Prediction

# In[34]:


def prediction():
    password = input("Enter the password: ")
    pass_array = np.array([password])
    pass_matrix = vectorizer.transform(pass_array)
    
    len_pass = len(password)
    len_normalized_lowercase = len([char for char in password if char.islower()])/len(password)
    new_matrix = np.append(pass_matrix.toarray(),(len_pass,len_normalized_lowercase)).reshape(1,101)
    result = clf.predict(new_matrix)
    
    if result == 0:
        return "Weak Password"
    elif result == 1:
        return "Normal Password"
    else:
        return "Strong Password"


# In[35]:


prediction()


# Checking the accuracy

# In[36]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy_score(y_test,y_pred)


# In[37]:


confusion_matrix(y_test,y_pred) 


# In[ ]:




