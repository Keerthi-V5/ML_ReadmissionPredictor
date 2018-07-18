
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


#After importing all the required packages above, reading the data from csv to dataframe in the following cell


# In[2]:


dataframe = pd.read_csv('C:/Users/yashas/LSProject/dataset_diabetes/diabetic_data.csv')


# In[3]:


#Number of rows and columns of dataframe
dataframe.shape


# In[4]:


#create concise table to understand the data

data_type = dataframe.dtypes.values
missing_count = dataframe.isnull().sum().values

unique_count = []
for attr in dataframe.columns:
    unique_count.append(dataframe[attr].unique().shape[0])

       
info ={'Attributes': dataframe.columns,
       'Attribute_Type': data_type,
       'MissingValue_Count':missing_count,
       'UniqueValue_Count': unique_count,
      
       }
col_names = {'Attributes','Attribute_Type','MissingValue_Count','UniqueValue_Count'}

info_tab = pd.DataFrame(info,columns = col_names)
info_tab


# In[6]:


#gives the distinct values in dataframe along with respective count
from collections import Counter
index=0
for col in dataframe.columns:
    if(index>1):
        print(col)
        print(Counter(dataframe[col]))
        print('************************************************************************************************************')
    else: index=index+1


# In[7]:


#Particularly for the target variable
dataframe['readmitted'].value_counts()


# In[8]:


#recoding 0 to No readmission and 1 to <30 or >30 readmission from readmitted attribute
dataframe['readmitted'] = pd.Series([0 if val == 'NO' 
                                     else 1
                                     for val in dataframe['readmitted']])


# In[9]:


#Representing target variable using bar chart
dataframe.groupby('readmitted').size().plot(kind='bar')
pt.ylabel('Count')


# In[10]:


#Representing age uaing bar chart
dataframe.groupby('age').size().plot(kind='bar')
pt.ylabel('Count')


# In[11]:


#With visual inspection we know that not all drugs show great variance 
#affecting the target variable. So consider few as examples and check

bar_graph = pt.figure(figsize=(20,20))

axis1 = bar_graph.add_subplot(221)
axis1 = dataframe.groupby('acarbose').size().plot(kind='bar')
pt.xlabel('acarbose', fontsize=15)
pt.ylabel('Count', fontsize=10)

axis2 = bar_graph.add_subplot(222)
axis2 = dataframe.groupby('citoglipton').size().plot(kind='bar')
pt.xlabel('citoglipton', fontsize=10)
pt.ylabel('Count', fontsize=10)

axis3 = bar_graph.add_subplot(223)
axis3 = dataframe.groupby('insulin').size().plot(kind='bar')
pt.xlabel('insulin', fontsize=10)
pt.ylabel('Count', fontsize=10)

axis4 = bar_graph.add_subplot(224)
axis4 = dataframe.groupby('metformin').size().plot(kind='bar')
pt.xlabel('metformin', fontsize=10)
pt.ylabel('Count', fontsize=10)


# In[ ]:


#hence drugs like acarbose which show no or least variance are not 
#considered whereas drugs like insulin, metformin which show variance are 
#considered


# In[18]:


#
a, axis = pt.subplots(10,8,figsize=(10,10))

features = ['age','gender','race']

index = 1

for f in features:
    pt.subplot(3,1,index)
    
    res = dataframe[[f,'readmitted']]
    sns.countplot(x="readmitted", hue=f, data=res, palette="Paired",edgecolor='white')
    pt.legend(loc=1, prop={'size':8})
    pt.title('Outcome based on '+features[index-1] )
    index = index+1
    
pt.tight_layout()
pt.show()

