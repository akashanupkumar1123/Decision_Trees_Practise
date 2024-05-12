#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
get_ipython().run_line_magic('matplotlib', 'inline')

print(np.__version__)
print(pd.__version__)
import sys
print(sys.version)
import sklearn
print(sklearn.__version__)


# In[22]:


from sklearn import tree


# In[24]:


X = [[0, 0], [1, 2]]
y = [0, 1]


# In[25]:


clf = tree.DecisionTreeClassifier()


# In[26]:


clf = clf.fit(X, y)


# In[27]:


clf.predict([[2., 2.]])


# In[28]:


clf.predict_proba([[2. , 2.]])


# In[29]:


clf.predict([[0.4, 1.2]])


# In[30]:


clf.predict_proba([[0.4, 1.2]])


# In[31]:


clf.predict_proba([[0, 0.2]])


# In[ ]:





# In[32]:


from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()


# In[33]:


iris.data[0:5]


# In[34]:


iris.feature_names


# In[35]:


X = iris.data[:, 2:]


# In[36]:


y = iris.target


# In[37]:



y


# In[38]:


clf = tree.DecisionTreeClassifier(random_state=42)


# In[39]:


clf = clf.fit(X, y)


# In[ ]:





# In[40]:


from sklearn.tree import export_graphviz


# In[41]:


export_graphviz(clf,
                out_file="tree.dot",
                feature_names=iris.feature_names[2:],
                class_names=iris.target_names,
                rounded=True,
                filled=True)


# In[ ]:





# In[42]:


import graphviz


# In[43]:


dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names[2:],
                                class_names=iris.target_names,
                                rounded=True,
                                filled=True)


# In[44]:


graph = graphviz.Source(dot_data)


# In[ ]:





# In[45]:


import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


df = sns.load_dataset('iris')
df.head()


# In[ ]:





# In[47]:


col = ['petal_length', 'petal_width']
X = df.loc[:, col]


# In[48]:


species_to_num = {'setosa': 0,
                  'versicolor': 1,
                  'virginica': 2}
df['tmp'] = df['species'].map(species_to_num)
y = df['tmp']


# In[49]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)


# In[50]:


Xv = X.values.reshape(-1,1)
h = 0.02
x_min, x_max = Xv.min(), Xv.max() + 1
y_min, y_max = y.min(), y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# In[ ]:





# In[51]:


z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
fig = plt.figure(figsize=(8,8))
ax = plt.contourf(xx, yy, z, cmap = 'afmhot', alpha=0.3);
plt.scatter(X.values[:, 0], X.values[:, 1], c=y, s=80, 
            alpha=0.9, edgecolors='g');


# In[ ]:





# In[52]:


def gini(p):
    return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))
def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))
def error(p):
    return 1 - np.max([p, 1 - p])


# In[53]:


x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]

sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]


# In[54]:


fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                   ['Entropy', 'Entropy (scaled)', 
                   'Gini Impurity', 
                   'Misclassification Error'],
                   ['-', '-', '--', '-.'],
                   ['black', 'lightgray',
                      'red', 'green', 'cyan']):
     line = ax.plot(x, i, label=lab, 
                    linestyle=ls, lw=2, color=c)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()


# In[ ]:





# In[55]:


from sklearn import tree


# In[56]:


X = [[0, 0], [3,3]]
y = [0.75, 3]


# In[57]:


tree_reg = tree.DecisionTreeRegressor(random_state=42)


# In[58]:


tree_reg = tree_reg.fit(X, y)


# In[59]:


tree_reg.predict([[1.5, 1.5]])


# In[ ]:





# In[60]:


# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure(figsize=(10,8))
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()


# In[ ]:





# In[61]:


# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure(figsize=(10,8))
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[62]:


from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()


# In[63]:


X = iris.data[:, 0:2]
y = iris.target
clf = tree.DecisionTreeClassifier(random_state=42)
clf = clf.fit(X, y)


# In[64]:


dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names[2:],
                                class_names=iris.target_names,
                                rounded=True,
                                filled=True)


# In[ ]:





# In[65]:


from sklearn.datasets import make_moons


# In[66]:


X_data, y_data = make_moons(n_samples=1000, noise=0.5, random_state=42)


# In[67]:


cl1 = tree.DecisionTreeClassifier(random_state=42)
cl2 = tree.DecisionTreeClassifier(min_samples_leaf=10, random_state=42)


# In[68]:


from sklearn.model_selection import train_test_split


# In[69]:


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)


# In[70]:


from sklearn.model_selection import GridSearchCV


# In[71]:


#params = {'max_leaf_nodes': list(range(2, 50)),
#          'min_samples_split': [2, 3, 4],
#          'min_samples_leaf': list(range(5, 20))}

params ={'min_samples_leaf': list(range(5, 20))}


# In[72]:


grid_search_cv = GridSearchCV(tree.DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1)

grid_search_cv.fit(X_train, y_train)


# In[73]:


grid_search_cv.best_estimator_


# In[74]:


from sklearn.metrics import accuracy_score


# In[75]:


y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)


# In[76]:


cl1.fit(X_train, y_train)
y_pred = cl1.predict(X_test)
accuracy_score(y_test, y_pred)


# In[77]:


cl1.get_params()


# In[ ]:




