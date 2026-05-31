#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
df= pd.read_csv(r"D:\mulit-text-classifier\data\train.csv")
df.head()
df['text']=df['text'].str.lower()


# In[4]:


df['text'][7]


# In[5]:


# remove punctuation
import string 
ex= string.punctuation
def remove_punch(text):
    return text.translate(str.maketrans("","",ex))


# In[6]:


df['text']= df['text'].apply(remove_punch)


# In[7]:


df['text'][7]


# In[10]:


df['labels'].unique()


# As we are working on hinglish problem as well, so i will noy apply stemming or lemmatization so that word doesnot change its meaning

# In[8]:


# now we will train and test
x= df['text']
y= df['labels']
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train,y_test= tts(x,y,test_size=0.2,random_state=42)


# In[7]:


x_train.head()


# In[8]:


y_train


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer= TfidfVectorizer(analyzer='char',ngram_range=(2,5),max_features=5000)
# we have used this because its more powerful 
# model learns automatically
# TF-IDF converts text into number


# In[10]:


x_train_trdf= vectorizer.fit_transform(x_train)
x_test_trdf= vectorizer.transform(x_test)
#You already converted the text into numbers using TF-IDF in Step 3.
# only for x not for y


# In[11]:


# label encoding is done only for y
from sklearn.preprocessing import LabelEncoder
encode= LabelEncoder()
y_train_en= encode.fit_transform(y_train)
y_test_en= encode.transform(y_test)


# In[12]:


from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(x_train_trdf,y_train_en)


# In[13]:


y_pred= model.predict(x_test_trdf)
from sklearn.metrics import accuracy_score, classification_report
# We CAN calculate metrics on train data, but we SHOULD evaluate mainly on test data.
print("Accuracy Score:", accuracy_score(y_test_en,y_pred))
print("Classfication Report",classification_report(y_test_en,y_pred)) # for comparsion


# In[14]:


# to save the model
import joblib


# In[15]:


joblib.dump(model,"language_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(encode,"label_encoder.pkl")

