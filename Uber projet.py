#!/usr/bin/env python
# coding: utf-8

#    <H1>UBER PROJECT<H1>

# # IMPORT LIBRARIES

# In[1]:


get_ipython().system('pip install plotly')


# In[2]:


import pandas as pd
import plotly.express as px
import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# # IMPORT DATASET

# In[3]:


uber_pickup_apr2014 = pd.read_csv("uber-raw-data-apr14.csv")
uber_pickup_apr2014.head()


# In[4]:


uber_pickup_may2014 = pd.read_csv("uber-raw-data-may14.csv")
uber_pickup_may2014.head()


# In[5]:


uber_pickup_jun2014 = pd.read_csv("uber-raw-data-jun14.csv")
uber_pickup_jun2014.head()


# In[6]:


uber_pickup_jul2014 = pd.read_csv("uber-raw-data-jul14.csv")
uber_pickup_jul2014.head()


# In[7]:


uber_pickup_aug2014 = pd.read_csv("uber-raw-data-aug14.csv")
uber_pickup_aug2014.head()


# In[8]:


uber_pickup_sep2014 = pd.read_csv("uber-raw-data-sep14.csv")
uber_pickup_sep2014.head()


# In[9]:


print(uber_pickup_apr2014.shape)
print(uber_pickup_may2014.shape)
print(uber_pickup_jun2014.shape)
print(uber_pickup_jul2014.shape)
print(uber_pickup_aug2014.shape)
print(uber_pickup_sep2014.shape)


# # READ AND EXPLORE THE DATASET

# In[10]:


dataset = pd.concat([uber_pickup_apr2014.sample(20000), uber_pickup_may2014.sample(20000), 
                  uber_pickup_jun2014.sample(20000), uber_pickup_jul2014.sample(20000), 
                  uber_pickup_aug2014.sample(20000), uber_pickup_sep2014.sample(20000)], 
                 axis=0)
dataset


# In[11]:


dataset.info()


# # DATA CLEANING

# - Column Date/Time

# In[12]:


dataset["Date/Time"]= pd.to_datetime(dataset["Date/Time"])
dataset = dataset.rename(columns={"Date/Time":"Date"})


# In[13]:


dataset.info()


# In[14]:


dataset["month"]= dataset.Date.dt.month
dataset["year"]= dataset.Date.dt.year
dataset["day"]= dataset.Date.dt.day
dataset["day_of_week"]= dataset.Date.dt.day_of_week
dataset["hour"]= dataset.Date.dt.hour
dataset = dataset.drop(["Date"], axis=1)
dataset


# # ANALYSE THE DATASET

# In[15]:


dataset.to_csv(r'Dataset_clean.csv', index = False)


# # PREPROCESSING

# In[16]:


X = dataset.loc[:, ["Lat", "Lon"]]
X


# In[17]:


X = pd.get_dummies(X, drop_first=True)
X


# # MACHINE LEARNING MODEL

# ## KMeans

# In[18]:


# Using the Elbow method to find the optimal number K of clusters

wcss =  []
for i in range (1,11): 
    kmeans = KMeans(n_clusters= i, init = "k-means++", random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    print(f"Iteration no: {i} Finished")


# In[19]:


# Create a DataFrame that will be fed to plotly 
wcss_frame = pd.DataFrame(wcss)


# Creating a line plot
fig = px.line(wcss_frame, x=wcss_frame.index, y=wcss_frame.iloc[:, -1])

# Creating layouts 
fig.update_layout(
    title="Inertia per clusters",
    xaxis_title="# clusters",
    yaxis_title="Inertia"
)

# Render in notebook
fig.show(renderer="notebook")


# In[20]:


# Computer mean silhouette score
sil = []

## Careful, you need to start at i=2 as silhouette score cannot accept less than 2 labels 
for i in range (2,11): 
    kmeans = KMeans(n_clusters= i, init = "k-means++", random_state = 0)
    kmeans.fit(X)
    sil.append(silhouette_score(X, kmeans.predict(X)))
    print("Silhouette score for K={} is {}".format(i, sil[-1]))


# In[21]:


# Instanciate KMeans with k=4 and initialisation with k-means++
# You should always use k-means++ as it alleviate the problem of local minimum convergence 
kmeans = KMeans(n_clusters=4, random_state=0, init="k-means++")

# Fit kmeans to our dataset
kmeans.fit(X)


# In[22]:


import numpy as np

X["cluster"] = kmeans.labels_
X.head()


# In[23]:


np.unique(kmeans.labels_)


# In[24]:


centroids_ = kmeans.cluster_centers_
centroids_


# In[25]:


dataset["cluster"]=X.cluster
dataset


# In[26]:


fig = px.scatter_mapbox(
        dataset, 
        lat="Lat", 
        lon="Lon",
        color="cluster",
        mapbox_style="open-street-map",
)

fig.show("notebook")


# In[29]:


import seaborn as sns

sns.catplot(data=dataset, y='cluster', kind='count', color=sns.color_palette()[0], order=dataset.cluster.value_counts().index);

