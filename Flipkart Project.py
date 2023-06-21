#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# In[2]:


import nltk
nltk.download()


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[5]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[6]:


data = pd.read_csv('flipkart_data.csv')
data.head()


# In[7]:


# unique ratings


# In[8]:


pd.unique(data['rating'])


# In[9]:


sns.countplot(data=data,
             x='rating',
             order=data.rating.value_counts().index)


# In[10]:


# rating label(final)


# In[11]:


pos_neg = []
for i in range(len(data['rating'])):
    if data['rating'][i] >= 5:
        pos_neg.append(1)
    else :
        pos_neg.append(0)

data['label'] = pos_neg


# In[12]:


from tqdm import tqdm

def preprocess_text(text_data):
    preprocessed_text = []
    
    for sentence in tqdm(text_data):
        #removing punctuations
        sentence = re.sub(r'[^\w\s]', '', sentence)
        
        #converting lowercase and removing stopwords
        preprocessed_text.append(' '.join(token.lower()
                                         for token in nltk.word_tokenize(sentence)
                                         if token.lower() not in stopwords.words('english')))
        
        return preprocessed_text


# In[13]:


data['new_column'] = pd.Series()


# In[15]:


data.head()


# In[16]:


data["label"].value_counts()


# In[17]:


consolidated = ' '.join(
    word for word in data['review'][data['label'] == 1].astype(str))
wordCloud = WordCloud(width=1600, height=800,
                     random_state=21, max_font_size=110)
plt.figure(figsize=(15, 10))
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
plt.axis('off')
plt.show()


# In[18]:


cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['review'] ).toarray()


# In[19]:


X


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, data['label'],
                                                    test_size=0.33,
                                                    stratify=data['label'],
                                                    random_state = 42)


# In[21]:


from sklearn.tree import DecisionTreeClassifier
  
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train,y_train)
  
#testing the model
pred = model.predict(X_train)
print(y_train,pred)


# In[22]:


from sklearn.metrics import confusion_matrix


# In[23]:


from sklearn import metrics
cm = confusion_matrix(y_train,pred)
  
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,
                                            display_labels = [False, True])
  
cm_display.plot()
plt.show()

