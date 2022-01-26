#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder


# In[2]:


df_1 = pd.read_csv('real_309_new_4.csv')
df_1.shape


# In[3]:


df_1.head(3)


# In[4]:


df_col = df_1.columns[df_1.dtypes == "object"]

counts_dict = {}

for col in df_col:
     counts_dict[col] = df_1[col].value_counts()
     df_1[col] = df_1[col].str.replace(r'[^\w\s]+', '')

df_2 = df_1[['id', 'label']]

df_2 = df_2.sort_values(by='label', axis=0, ascending=True)

le = LabelEncoder()

df_2['label_index'] = le.fit_transform(df_2['label'])
df_2[['label', 'label_index']].value_counts()
group = df_2.groupby('label')
money_group = group.first()

df_3 = df_1.drop(['label'], axis=1)

df3_col = df_3.columns[df_3.dtypes == "object"]
    
for col in df3_col:
    df_3[col] = le.fit_transform(df_3[col].astype(str))
    
result = pd.merge(df_3, df_2, how = 'left', on = ['id'])
result = result.drop(['id', 'label'], axis=1)

result = result.rename(columns = {'label_index': 'label'})

result.to_csv('real_309_new_4_en.csv', index=False, sep=',')


result['label'].value_counts()


# In[ ]:


df5 = pd.read_csv('real_310_4_en.csv')


# In[ ]:


df5


# In[ ]:




