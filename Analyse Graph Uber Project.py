#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[1]:


get_ipython().system('pip install plotly')


# In[2]:


import plotly.express as px
import pandas as pd


# # READ DATASET

# In[3]:


df = pd.read_csv('Dataset_clean.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# ### Hot zone per day_of_week

# In[6]:


per_day_of_week=pd.DataFrame(df['day_of_week'].value_counts()).reset_index()
per_day_of_week.columns=['day_of_week','Count']
per_day_of_week=per_day_of_week.sort_values(by='day_of_week')
per_day_of_week


# In[7]:


fig = px.bar(data_frame=per_day_of_week, x="day_of_week", y="Count", color="day_of_week")
fig.show()


# ### Hot zone per hour

# In[8]:


per_hour=pd.DataFrame(df['hour'].value_counts()).reset_index()
per_hour.columns=['hour','Count']
per_hour=per_hour.sort_values(by='hour')
per_hour


# In[9]:


fig = px.bar(data_frame=per_hour, x="hour", y="Count", color="hour")
fig.show()

