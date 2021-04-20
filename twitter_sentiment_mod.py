#!/usr/bin/env python
# coding: utf-8

# In[50]:


import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


# In[51]:


def mode_function(x):
    k=nltk.FreqDist(x)   
    return list(k.keys())[0]


# In[52]:


class voteclassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers
        
    def classify(self,features):
        votes=[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return (mode_function(votes))
    
    def confidence(self,features):
        votes=[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode_function(votes)) ## how many time the mode value has appeared in the function
        conf = choice_votes / len(votes)
        return conf
    


# In[53]:


documents_f = open("documents_twitter.pickle","rb")
documents  = pickle.load(documents_f)
documents_f.close()


# In[54]:


word_features_5 = open("word_features.pickle",'rb')
word_features = pickle.load(word_features_5)
word_features_5.close()


# In[55]:


def find_features(document):
        # words=set(document)  ## you cannot use this, since the input is now sring and not words as seperate
        # but the above will work but with less accurcay
        
    words = word_tokenize(document)
    features={}   ## created a dictionary
    for w in word_features:
        features[w]=(w in words)  ## will set the value a s 1 if there else 0
        ## will create a boolean value for each word
    
    return features


# In[56]:


featureset_5 = open("featuresets_twitter.pickle",'rb')
featuresets = pickle.load(featureset_5)
featureset_5.close()


# In[57]:


random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[10000:]
training_set = featuresets[:10000]



open_file = open("originalnaivebayes_twitter.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("MNB_classifier_twitter.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()



open_file = open("BernoulliNB_classifier_twitter.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("LogisticRegression_classifier_twitter.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("LinearSVC_classifier_twitter.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


open_file = open("SGDC_classifier_twitter.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()


# In[58]:



voted_classifier = voteclassifier(classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


# In[59]:


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)


# In[60]:


## just save the file
## and use the same file name in other script so as to run all the programs


# In[ ]:





# In[61]:


import os
os.getcwd()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




