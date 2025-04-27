#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[4]:


get_ipython().system('pip install textblob')


# In[5]:


from textblob import TextBlob


# In[6]:


#regular expression
import re


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


from sklearn.ensemble import RandomForestClassifier


# In[9]:


from sklearn.metrics import classification_report,accuracy_score


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


import seaborn as sns


# In[13]:


get_ipython().system('pip install shap')


# In[14]:


import shap


# In[15]:


data = {
    'Employee_ID': ['E001', 'E002', 'E003', 'E004', 'E005', 'E006', 'E007', 'E008', 'E009', 'E010'],
    'Email_Text': [
        "I feel overwhelmed with the workload and I'm unsure about the deadlines.",
        "Everything is going well, looking forward to the team meeting tomorrow.",
        "I'm finding it hard to manage everything, I need more time.",
        "I'm happy with the tasks and the project, everything is manageable.",
        "Not sure how long I can keep up with all these deadlines, feeling exhausted.",
        "The project is going well, but I wish we had more resources.",
        "I'm a bit stressed about the workload, but trying to manage.",
        "Everything seems to be going fine, but the pressure is building.",
        "I need more time to finish everything. It's overwhelming.",
        "Everything is okay for now, but I feel a little disconnected."
    ],
    'Attrition_Label': ['At Risk', 'Stable', 'At Risk', 'Stable', 'At Risk', 'Stable', 'At Risk', 'Stable', 'At Risk', 'Stable']
}


# In[16]:


df=pd.DataFrame(data)


# In[17]:


df.head()


# In[18]:


def clean_text(text):
    text=re.sub(r'\s+',' ',text)
    text=re.sub(r'[^\w\s]','',text)
    text=text.lower()
    return text


# In[20]:


df['Cleaned_Email']=df['Email_Text'].apply(clean_text)


# In[21]:


def get_sentiment(text):
    return TextBlob(text).sentiment.polarity


# In[31]:


def get_word_ratios(text):
    positive_words = ["good", "great", "fantastic", "excited", "happy", "satisfied"]
    negative_words = ["stress", "overwhelmed", "disappointed", "unhappy", "exhausted", "angry"]

    positive_count = sum(1 for word in text.split() if word in positive_words)
    negative_count = sum(1 for word in text.split() if word in negative_words)
    total_count = len(text.split())

    positive_ratio = positive_count / total_count if total_count != 0 else 0
    negative_ratio = negative_count / total_count if total_count != 0 else 0

    return positive_ratio, negative_ratio


# In[32]:


df['Sentiment_polarity']=df['Cleaned_Email'].apply(get_sentiment)


# In[33]:


df['postive_word_ratio'],df['negative_word_ratio']=zip(*df['Cleaned_Email'].apply(get_word_ratios))


# In[34]:


df['word_count']=df['Cleaned_Email'].apply(lambda x: sum(1 for word in x.split()))


# In[35]:


stress_terms=['stressed','overwhelmed','exahausted','burnout']


# In[36]:


df['Stress_Term_Frequency']=df['Cleaned_Email'].apply(lambda x: sum(1 for word in x.split() if word in stress_terms))


# In[37]:


df.head()


# In[38]:


df["Attrition_Label"]=df['Attrition_Label'].map({'At Risk':1,'Stable':0})


# In[39]:


df.head()


# In[42]:


x=df[['Sentiment_polarity','postive_word_ratio','negative_word_ratio','word_count','Stress_Term_Frequency']]


# In[44]:


y=df['Attrition_Label']


# In[45]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[46]:


x_train.head()


# In[47]:


model=RandomForestClassifier(n_estimators=100,random_state=42)


# In[48]:


model.fit(x_train,y_train)


# In[49]:


y_pred=model.predict(x_test)


# In[51]:


print(y_pred)


# In[52]:


accuracy=accuracy_score(y_test,y_pred)


# In[53]:


print(accuracy*100)


# In[54]:


print(classification_report(y_test,y_pred))


# In[55]:


explainer = shap.TreeExplainer(model)


# In[57]:


shap_values = explainer.shap_values(x_test)


# In[59]:


shap.summary_plot(shap_values, x_test)


# In[80]:


sample_email = "I feel happy and excited."


# In[81]:


sample_email_cleaned=clean_text(sample_email)


# In[82]:


print(sample_email_cleaned)


# In[83]:


sample_sentiment=get_sentiment(sample_email_cleaned)


# In[84]:


sample_positive_ratio, sample_negative_ratio = get_word_ratios(sample_email_cleaned)


# In[85]:


sample_word_count = len(sample_email_cleaned.split())


# In[86]:


sample_stress_term_freq = sum(1 for word in sample_email_cleaned.split() if word in stress_terms)


# In[87]:


sample_data = pd.DataFrame([[sample_sentiment, sample_positive_ratio, sample_negative_ratio, sample_word_count, sample_stress_term_freq]],
                           columns=['Sentiment_Polarity', 'Positive_Word_Ratio', 'Negative_Word_Ratio', 'Word_Count', 'Stress_Term_Frequency'])


# In[88]:


sample_data.head()


# In[89]:


sample_data = pd.DataFrame([[sample_sentiment, sample_positive_ratio, sample_negative_ratio, sample_word_count, sample_stress_term_freq]],
                           columns=['Sentiment_polarity', 'postive_word_ratio', 'negative_word_ratio', 'word_count', 'Stress_Term_Frequency'])

# Ensure the column names match the training data exactly
sample_data = sample_data.rename(columns={
    'postive_word_ratio': 'postive_word_ratio',  # No change
    'negative_word_ratio': 'negative_word_ratio',  # No change
    'word_count': 'word_count',  # No change
    'Sentiment_polarity': 'Sentiment_polarity'  # No change
})


# In[90]:


sample_prediction = model.predict(sample_data)
print(sample_prediction)


# In[91]:


plt.figure(figsize=(8, 6))
sns.histplot(df['Sentiment_polarity'], kde=True, bins=10)
plt.title('Distribution of Sentiment Polarity')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()


# In[ ]:




