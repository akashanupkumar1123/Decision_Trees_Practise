#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[2]:


df = pd.read_csv('kyphosis.csv')


# In[3]:


df.head()


# In[ ]:





# In[4]:


sns.pairplot(df,hue='Kyphosis',palette='Set1')


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[ ]:





# In[8]:


from sklearn.tree import DecisionTreeClassifier


# In[9]:


dtree = DecisionTreeClassifier()


# In[10]:


dtree.fit(X_train,y_train)


# In[ ]:





# In[11]:


predictions = dtree.predict(X_test)


# In[12]:


from sklearn.metrics import classification_report,confusion_matrix


# In[13]:


print(classification_report(y_test,predictions))


# In[ ]:





# In[14]:


print(confusion_matrix(y_test,predictions))


# In[21]:


from sklearn.tree import plot_tree


# In[25]:


plt.figure(figsize=(30,18))
plot_tree(dtree);


# In[26]:


from IPython.display import Image  
from io import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(df.columns[1:])
features


# In[ ]:





# In[27]:


dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())  


# In[ ]:




