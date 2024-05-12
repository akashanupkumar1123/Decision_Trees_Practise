#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("penguins_size.csv")


# In[3]:


df.head()


# In[ ]:





# In[4]:


df.info()


# In[ ]:





# In[5]:


df.isna().sum()


# In[ ]:





# In[6]:


df.isna().sum()


# In[ ]:





# In[7]:


df = df.dropna()


# In[ ]:





# In[8]:


df.info()


# In[ ]:





# In[9]:


df.head()


# In[ ]:





# In[10]:


df['sex'].unique()


# In[ ]:





# In[11]:


df['island'].unique()


# In[ ]:





# In[12]:


df = df[df['sex']!='.']


# In[ ]:





# In[13]:


sns.scatterplot(x='culmen_length_mm',y='culmen_depth_mm',data=df,hue='species',palette='Dark2')


# In[ ]:





# In[14]:


sns.pairplot(df,hue='species',palette='Dark2')


# In[ ]:





# In[15]:


sns.catplot(x='species',y='culmen_length_mm',data=df,kind='box',col='sex',palette='Dark2')


# In[ ]:





# In[16]:


pd.get_dummies(df)


# In[ ]:





# In[17]:


pd.get_dummies(df.drop('species',axis=1),drop_first=True)


# In[ ]:





# In[18]:


X = pd.get_dummies(df.drop('species',axis=1),drop_first=True)
y = df['species']


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:





# In[21]:


from sklearn.tree import DecisionTreeClassifier


# In[22]:


model = DecisionTreeClassifier()


# In[23]:


model.fit(X_train,y_train)


# In[24]:


base_pred = model.predict(X_test)


# In[ ]:





# In[25]:


from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix


# In[ ]:





# In[26]:


confusion_matrix(y_test,base_pred)


# In[ ]:





# In[27]:


plot_confusion_matrix(model,X_test,y_test)


# In[ ]:





# In[28]:


print(classification_report(y_test,base_pred))


# In[ ]:





# In[29]:


model.feature_importances_


# In[ ]:





# In[30]:


pd.DataFrame(index=X.columns,data=model.feature_importances_,columns=['Feature Importance'])


# In[ ]:





# In[31]:


sns.boxplot(x='species',y='body_mass_g',data=df)


# In[ ]:





# In[ ]:





# In[32]:


from sklearn.tree import plot_tree


# In[33]:


plt.figure(figsize=(12,8))
plot_tree(model);


# In[ ]:





# In[34]:


plt.figure(figsize=(12,8),dpi=150)
plot_tree(model,filled=True,feature_names=X.columns);


# In[ ]:





# In[35]:


def report_model(model):
    model_preds = model.predict(X_test)
    print(classification_report(y_test,model_preds))
    print('\n')
    plt.figure(figsize=(12,8),dpi=150)
    plot_tree(model,filled=True,feature_names=X.columns);


# In[ ]:





# In[36]:


help(DecisionTreeClassifier)


# In[ ]:





# In[37]:


pruned_tree = DecisionTreeClassifier(max_depth=2)
pruned_tree.fit(X_train,y_train)


# In[ ]:





# In[38]:


pruned_tree = DecisionTreeClassifier(max_depth=2)
pruned_tree.fit(X_train,y_train)


# In[ ]:





# In[39]:


pruned_tree = DecisionTreeClassifier(max_leaf_nodes=3)
pruned_tree.fit(X_train,y_train)


# In[ ]:





# In[40]:


report_model(pruned_tree)


# In[ ]:





# In[41]:


entropy_tree = DecisionTreeClassifier(criterion='entropy')
entropy_tree.fit(X_train,y_train)


# In[ ]:





# In[42]:


report_model(entropy_tree)


# In[ ]:




