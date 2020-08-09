#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#import re
import nltk
from nltk.tokenize import word_tokenize 
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# In[2]:


data=pd.read_csv('/home/dimas/Desktop/data/Book4.csv', encoding='cp1251')
data = data.astype('str')
data.dropna()
data.describe()


# In[3]:


data=data.sample(frac=0.8, replace=False, random_state=7)
#data.describe()
data.head()


# In[3]:


data.reset_index(drop=True, inplace=True)


# In[5]:


def tokenize(text):
   # text = re.sub(r'\W+|\d+|_', ' ', text)              
    tokens = nltk.word_tokenize(str(text))                            
    tokens = [word[-3:] for word in tokens] 
    tokens = " ".join(tokens)
    tokens = tokens.upper()# 5   
    return tokens


# In[6]:


data['n_tokens'] = data['n'].apply(tokenize)
print(data.shape)


# In[7]:


data


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
countvec1 = CountVectorizer(lowercase=False)
mv = pd.DataFrame(countvec1.fit_transform(data['n_tokens']).toarray(), columns=countvec1.get_feature_names(), index=None)
mv['class'] = data['p']
mv['name'] = data['n']
print(mv.shape)
mv.head()


# In[11]:


X = mv[mv.columns[:-2]]
y = mv['class']

y=y.values
y = y.ravel()
y=LabelEncoder().fit_transform(y)
print(X.shape)
print(y.shape)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(loss='log')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred_prob = classifier.predict_proba(X_test)



print (classifier,
       "\nBasic predictor:\n",
       max(y_test.mean(), 1-y_test.mean()),
       "\nAccuracy score:\n", 
       metrics.accuracy_score(y_test, y_pred),
       "\nConfusion matrix:\n", 
       metrics.confusion_matrix(y_test, y_pred),
       "\nClassification report:\n",
       metrics.classification_report(y_test, y_pred))

pos_prob = y_pred_prob[:, 1]

auc_1 = metrics.roc_auc_score(y_test, pos_prob)

print('ROC_AUC_1:', auc_1 )
fpr, tpr, thresholds = metrics.roc_curve(y_test, pos_prob)

plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[20]:





#defining parameter range 
param_grid = {'loss':['log'], 
              'penalty':["l2"],
              'epsilon':[0.1],
              'alpha' :[0.000001 ],#0.0001,
              'max_iter':[5,10,20,30,50,100,500,1000,2000]}

roc_auc_scorer = metrics.make_scorer(metrics.roc_auc_score, greater_is_better=True, needs_proba= True)
                                   #  needs_threshold=True)



grid = GridSearchCV(SGDClassifier(), param_grid, refit = True, verbose = 3, scoring = roc_auc_scorer) #scoring = 'accuracy'‘roc_auc’
  
# fitting the model for grid search 
grid.fit(X_train, y_train) 

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 

#grid_predictions = grid.predict(X_test) 
  
#print(classification report)
#print(metrics.classification_report(y_test, grid_predictions)) 
#print(metrics.roc_auc_score(y_test, grid_predictions))


# In[23]:


classifier = grid.best_estimator_
#classifier = SGDClassifier(alpha=1e-06, average=False, class_weight=None,
#          early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
#          l1_ratio=0.15, learning_rate='optimal', loss='log', max_iter=1000,
#          n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,
#          random_state=None, shuffle=True, tol=0.001,
#          validation_fraction=0.1, verbose=0, warm_start=False)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred_prob = classifier.predict_proba(X_test)



print (classifier,
       "\nBasic predictor:\n",
       max(y_test.mean(), 1-y_test.mean()),
       "\nAccuracy score:\n", 
       metrics.accuracy_score(y_test, y_pred),
       "\nConfusion matrix:\n", 
       metrics.confusion_matrix(y_test, y_pred),
       "\nClassification report:\n",
       metrics.classification_report(y_test, y_pred))

pos_prob = y_pred_prob[:, 1]

auc_2 = metrics.roc_auc_score(y_test, pos_prob)

print('ROC_AUC:', auc_2 )
fpr, tpr, thresholds = metrics.roc_curve(y_test, pos_prob)

plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[25]:


print(auc_2-auc_1)


# In[ ]:




