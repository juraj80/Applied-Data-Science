
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # The Series Data Structure

# In[1]:

import pandas as pd
animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)


# In[2]:

numbers = [1, 2, 3]
pd.Series(numbers)


# In[3]:

animals = ['Tiger', 'Bear', None]
pd.Series(animals)


# In[4]:

numbers = [1, 2, None]
pd.Series(numbers)


# In[5]:

import numpy as np
np.nan == None


# In[6]:

np.nan == np.nan


# In[7]:

np.isnan(np.nan)


# In[8]:

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s


# In[9]:

s.index


# In[10]:

s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
s


# In[11]:

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
s


# # Querying a Series

# In[12]:

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s


# In[13]:

s.iloc[3]


# In[14]:

s.loc['Golf']


# In[15]:

s[3]


# In[16]:

s['Golf']


# In[17]:

sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
s = pd.Series(sports)


# In[18]:

s[0] #This won't call s.iloc[0] as one might expect, it generates an error instead


# In[19]:

s = pd.Series([100.00, 120.00, 101.00, 3.00])
s


# In[20]:

total = 0
for item in s:
    total+=item
print(total)


# In[21]:

import numpy as np

total = np.sum(s)
print(total)


# In[22]:

#this creates a big series of random numbers
s = pd.Series(np.random.randint(0,1000,10000))
s.head()


# In[23]:

len(s)


# In[24]:

get_ipython().run_cell_magic('timeit', '-n 100', 'summary = 0\nfor item in s:\n    summary+=item')


# In[25]:

get_ipython().run_cell_magic('timeit', '-n 100', 'summary = np.sum(s)')


# In[26]:

s+=2 #adds two to each item in s using broadcasting
s.head()


# In[27]:

for label, value in s.iteritems():
    s.set_value(label, value+2)
s.head()


# In[28]:

get_ipython().run_cell_magic('timeit', '-n 10', 's = pd.Series(np.random.randint(0,1000,10000))\nfor label, value in s.iteritems():\n    s.loc[label]= value+2')


# In[29]:

get_ipython().run_cell_magic('timeit', '-n 10', 's = pd.Series(np.random.randint(0,1000,10000))\ns+=2')


# In[30]:

s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'
s


# In[31]:

original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'], 
                                   index=['Cricket',
                                          'Cricket',
                                          'Cricket',
                                          'Cricket'])
all_countries = original_sports.append(cricket_loving_countries)


# In[32]:

original_sports


# In[33]:

cricket_loving_countries


# In[34]:

all_countries


# In[35]:

all_countries.loc['Cricket']


# # The DataFrame Data Structure

# In[36]:

import pandas as pd
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()


# In[37]:

df.loc['Store 2']


# In[38]:

type(df.loc['Store 2'])


# In[39]:

df.loc['Store 1']


# In[40]:

df.loc['Store 1', 'Cost']


# In[41]:

df.T


# In[42]:

df.T.loc['Cost']


# In[43]:

df['Cost']


# In[44]:

df.loc['Store 1']['Cost']


# In[45]:

df.loc[:,['Name', 'Cost']]


# In[46]:

df.drop('Store 1')


# In[47]:

df


# In[48]:

copy_df = df.copy()
copy_df = copy_df.drop('Store 1')
copy_df


# In[49]:

get_ipython().magic('pinfo copy_df.drop')


# In[50]:

del copy_df['Name']
copy_df


# In[51]:

df['Location'] = None
df


# # Dataframe Indexing and Loading

# In[52]:

costs = df['Cost']
costs


# In[53]:

costs+=2
costs


# In[54]:

df


# In[55]:

get_ipython().system('cat olympics.csv')


# In[56]:

df = pd.read_csv('olympics.csv')
df.head()


# In[57]:

df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)
df.head()


# In[58]:

df.columns


# In[59]:

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#' + col[1:]}, inplace=True) 

df.head()


# # Querying a DataFrame

# In[60]:

df['Gold'] > 0


# In[61]:

only_gold = df.where(df['Gold'] > 0)
only_gold.head()


# In[62]:

only_gold['Gold'].count()


# In[63]:

df['Gold'].count()


# In[64]:

only_gold = only_gold.dropna()
only_gold.head()


# In[65]:

only_gold = df[df['Gold'] > 0]
only_gold.head()


# In[66]:

len(df[(df['Gold'] > 0) | (df['Gold.1'] > 0)])


# In[67]:

df[(df['Gold.1'] > 0) & (df['Gold'] == 0)]


# # Indexing Dataframes

# In[68]:

df.head()


# In[69]:

df['country'] = df.index
df = df.set_index('Gold')
df.head()


# In[70]:

df = df.reset_index()
df.head()


# In[71]:

df = pd.read_csv('census.csv')
df.head()


# In[72]:

df['SUMLEV'].unique()


# In[73]:

df=df[df['SUMLEV'] == 50]
df.head()


# In[74]:

columns_to_keep = ['STNAME',
                   'CTYNAME',
                   'BIRTHS2010',
                   'BIRTHS2011',
                   'BIRTHS2012',
                   'BIRTHS2013',
                   'BIRTHS2014',
                   'BIRTHS2015',
                   'POPESTIMATE2010',
                   'POPESTIMATE2011',
                   'POPESTIMATE2012',
                   'POPESTIMATE2013',
                   'POPESTIMATE2014',
                   'POPESTIMATE2015']
df = df[columns_to_keep]
df.head()


# In[75]:

df = df.set_index(['STNAME', 'CTYNAME'])
df.head()


# In[76]:

df.loc['Michigan', 'Washtenaw County']


# In[77]:

df.loc[ [('Michigan', 'Washtenaw County'),
         ('Michigan', 'Wayne County')] ]


# # Missing values

# In[78]:

df = pd.read_csv('log.csv')
df


# In[79]:

import pandas as pd
get_ipython().magic('pinfo pd.Series')


# In[80]:

get_ipython().magic('pinfo df.fillna')


# In[81]:

df = df.set_index('time')
df = df.sort_index()
df


# In[82]:

df = df.reset_index()
df = df.set_index(['time', 'user'])
df


# In[83]:

df = df.fillna(method='ffill')
df.head()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



