#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import mlflow
#import mlflow.sklearn


# # OBTENIR LE DATASET

# In[3]:


df = pd.read_csv('pointure.data')
df


# # EXPLORATION DES DONNÉES

# In[4]:


df.columns


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.describe()


# # PRE-TRAITEMENT DES DONNÉES

# In[8]:


import numpy as np
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
input_classes = ['masculin','féminin']
label_encoder.fit(input_classes)

# transformer un ensemble de classes
encoded_labels = label_encoder.transform(df['Genre'])
print(encoded_labels)
df['Genre'] = encoded_labels

df


# In[9]:


df.plot()


# In[10]:


df['Genre'].diff().hist(color='k', alpha=0.5, bins=50)


# In[11]:


df['Taille(cm)'].diff().hist(color='k', alpha=0.5, bins=50)


# In[12]:


df['Poids(kg)'].diff().hist(color='k', alpha=0.5, bins=50)


# In[13]:


df['Pointure(cm)'].diff().hist(color='k', alpha=0.5, bins=50)


# In[14]:


dfplot = pd.DataFrame(df.iloc[:, lambda dfToPredict: [0, 1, 2, 3]], columns=['Genre', 'Taille(cm)', 'Poids(kg)', 'Pointure(cm)'])
dfplot.diff().hist(color='k', alpha=0.5, bins=50)


# In[15]:


color = {'boxes': 'DarkGreen', 'whiskers': 'DarkOrange', 'medians': 'DarkBlue', 'caps': 'Gray'}
dfplot.plot.box(color=color, sym='r+')


# # MATRICE DE CORRELATION ET DE PERSON

# In[16]:


from pandas.plotting import scatter_matrix
scatter_matrix(dfplot, alpha=0.2, figsize=(6, 6), diagonal='kde')


# In[17]:


sns.pairplot(dfplot, diag_kind='kde', dropna=True)


# In[18]:


corr = dfplot.corr()
corr = corr.round(3)
f, ax = plt.subplots(figsize=(16, 12))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
_ = sns.heatmap(corr, cmap="YlGn", square=True, ax = ax, annot=True, linewidth = 0.1)
plt.title('Corrélation de Pearson', y=1.05, size=15)
plt.show()


# # DEFINIR LES FEATURES
# # SEPARER LE DATASET EN TRAIN ET TEST

# In[19]:


X = df.iloc[:, lambda df: [1, 2, 3]]
y = df.iloc[:, 0]


# In[20]:


from sklearn.model_selection import train_test_split

#decomposer les donnees predicteurs en training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)


# In[21]:


print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# # FAIRE APPRENDRE LE MODELE

# In[22]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)


# # EVALUATION SUR LE TRAIN

# In[23]:


y_naive_bayes1 = gnb.predict(X_train)
print("Number of mislabeled points out of a total 0%d points : 0%d" % (X_train.shape[0],(y_train != y_naive_bayes1).sum()))


# In[24]:


from sklearn import metrics
accuracy = metrics.accuracy_score(y_train, y_naive_bayes1)
print("Accuracy du modele Naive Bayes predit: " + str(accuracy))


recall_score = metrics.recall_score(y_train, y_naive_bayes1)
print("recall score du modele Naive Bayes predit: " + str(recall_score))

f1_score = metrics.f1_score(y_train, y_naive_bayes1)
print("F1 score du modele Naive Bayes predit: " + str(f1_score))


# # EVALUATION SUR LE TEST

# In[25]:


y_naive_bayes2 = gnb.predict(X_test)
print("Number of mislabeled points out of a total 0%d points : 0%d" % (X_test.shape[0],(y_test != y_naive_bayes2).sum()))

recall_score = metrics.recall_score(y_test, y_naive_bayes2)
print("recall score du modele Naive Bayes predit: " + str(recall_score))

f1_score = metrics.f1_score(y_test, y_naive_bayes2)
print("F1 score du modele Naive Bayes predit: " + str(f1_score))


# # PREDICTION SUR UNE OBSERVATION

# In[26]:


d = {'Taille(cm)':[183], 'Poids(kg)':[59], 'Pointure(cm)':[20]}
dfToPredict = pd.DataFrame(data=d) 
dfToPredict


# In[27]:


yPredict = gnb.predict(dfToPredict)
print('La classe predite est : ', yPredict)


# ## MLFlow: Metrics Monitoring

# In[28]:


from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

d = {'Taille(cm)':[183], 'Poids(kg)':[59], 'Pointure(cm)':[20]}
dfToPredict = pd.DataFrame(data=d) 
dfToPredict


## Training 
gnb = GaussianNB()
gnb.fit(X_train, y_train)
    
y_naive_bayes1 = gnb.predict(X_train)
    
accuracy1 = metrics.accuracy_score(y_train, y_naive_bayes1)
print("Training, Accuracy du modele Naive Bayes predit: " + str(accuracy))
    
recall_score1 = metrics.recall_score(y_train, y_naive_bayes1)
print("recall score du modele Naive Bayes predit: " + str(recall_score))
    
f1_score1 = metrics.f1_score(y_train, y_naive_bayes1)
print("F1 score du modele Naive Bayes predit: " + str(f1_score))
    
    
## Testing
    
y_naive_bayes2 = gnb.predict(X_test)
print("Testing , Number of mislabeled points out of a total 0%d points : 0%d" % (X_test.shape[0],(y_test != y_naive_bayes2).sum()))
    
recall_score2 = metrics.recall_score(y_test, y_naive_bayes2)
print("recall score du modele Naive Bayes 2 predit: " + str(recall_score))
    
f1_score2 = metrics.f1_score(y_test, y_naive_bayes2)
print("F1 score du modele Naive Bayes 2 predit: " + str(f1_score))
    

# Classe predite

yPredict = gnb.predict(dfToPredict)
print('La classe predite est : ', yPredict)
    

# In[31]:


# Write scores to a file
with open("metrics.txt", 'w') as outfile:
    # Training Metrics
    outfile.write("Accuracy training:  {0:2.1f} \n".format(accuracy1))
    outfile.write("recall score training: {0:2.1f} \n".format(recall_score1))
    outfile.write("f1_score training:  {0:2.1f} \n".format(f1_score1))
    
    # Testing Metrics
    outfile.write("recall score test:  {0:2.1f} \n".format(recall_score2))
    outfile.write("f1_score test: {0:2.1f} \n".format(f1_score2))
    
    # Classe predite
    outfile.write('Predict Class: {0:2.1f} \n'.format(yPredict[0]))


# In[ ]:
