#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('dataset.csv')


# In[3]:


df.isnull().sum()


# In[4]:


df.head()


# In[5]:


df.corr()


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


from sklearn.model_selection import cross_val_score


# In[13]:


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
score=cross_val_score(classifier,df.drop("target",axis=1), df.target,cv=10)


# In[14]:


score


# In[15]:


score.mean()


# In[16]:


## Apply Algorithm

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)


# In[17]:


score=cross_val_score(random_forest_model,df.drop("target",axis=1), df.target,cv=10)


# In[18]:


score


# In[19]:


score.mean()


# In[20]:


## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


# In[21]:


## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost


# In[22]:


classifier=xgboost.XGBClassifier()


# In[23]:


random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[24]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[26]:


from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(df.drop("target",axis=1), df.target)
timer(start_time) # timing ends here for "start_time" var


# In[27]:


random_search.best_estimator_


# In[28]:


# cross validation for random forest
classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.2,
              learning_rate=0.1, max_delta_step=0, max_depth=4,
              min_child_weight=7, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)


# In[29]:


score=cross_val_score(classifier,df.drop("target",axis=1), df.target,cv=10)


# In[30]:


score


# In[31]:


score.mean()


# In[ ]:


#xgboost is giving best accuracy

