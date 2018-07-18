
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import seaborn as sb


# In[2]:


dataframe1 = pd.read_csv('C:/Users/yashas/LS_Dataset/diabetic_data.csv')


# In[3]:


print dataframe1.columns.values
#dataframe1.head(5)
dataframe1.count


# In[4]:


#After many iterations these attributes are being removed
#unique ids, won't contribute to classification

del dataframe1['encounter_id']
del dataframe1['patient_nbr']
del dataframe1['payer_code']

#Higher percentage of missing values
del dataframe1['weight']
del dataframe1['medical_specialty']

#These won't affect the classification as the variance is too low
del dataframe1['diag_2']
del dataframe1['diag_3']

del dataframe1['repaglinide']
del dataframe1['nateglinide']
del dataframe1['chlorpropamide']
del dataframe1['acetohexamide']
del dataframe1['tolbutamide']
del dataframe1['acarbose']
del dataframe1['miglitol']
del dataframe1['troglitazone']
del dataframe1['tolazamide']
del dataframe1['examide']
del dataframe1['citoglipton']
del dataframe1['glyburide-metformin']
del dataframe1['glipizide-metformin']
del dataframe1['glimepiride-pioglitazone']
del dataframe1['metformin-rosiglitazone']
del dataframe1['metformin-pioglitazone']

print dataframe1.columns.values


# In[5]:


#recoding 0 to No readmission and 1 to <30 or >30 readmission from readmitted attribute
dataframe1['readmitted'] = pd.Series([0 if val == 'NO' 
                                     else 1
                                     for val in dataframe1['readmitted']])


# In[6]:


#Direct binary cnversion
dataframe1['change'] = pd.Series([0 if val == 'No' 
                                     else 1
                                     for val in dataframe1['change']])
dataframe1['diabetesMed'] = pd.Series([0 if val == 'No' 
                                     else 1
                                     for val in dataframe1['diabetesMed']])

#print(dataframe1['readmitted'])
#'change','diabetesMed',


# In[7]:


dataframe1.shape


# In[ ]:


#dataframe1['race'].replace('Unknown','?',inplace=True)


# In[ ]:


#not consider rows having ? in dataframe or removing empty rows in race and diag_1

#dataframe1 = dataframe1[dataframe1['race']!= '?']
#dataframe1 = dataframe1[dataframe1['diag_1']!= '?']

#dataframe1[17:23]


# In[ ]:


#dataframe1.shape


# In[8]:


#Making categorical attribute to binary with 1 for diabetic category and 0 for others
dataframe1['diag_1'] = pd.Series([1 if val.startswith('250')
                                 else 0
                                 for val in dataframe1['diag_1']], index = dataframe1.index)

#dataframe1['diag_1']


# In[9]:


#Square root transformation to reduce effects of extreme values
dataframe1['number_outpatient'] = dataframe1['number_outpatient'].apply(lambda x:np.sqrt(x+0.5))
dataframe1['number_inpatient'] = dataframe1['number_inpatient'].apply(lambda x:np.sqrt(x+0.5))
dataframe1['number_emergency'] = dataframe1['number_emergency'].apply(lambda x:np.sqrt(x+0.5))

#dataframe1['number_inpatient']


# In[10]:


#standardisation to have 0 mean and unit variance
scale_attr = ['time_in_hospital',
             'num_lab_procedures','num_procedures','num_medications',
             'number_outpatient','number_emergency','number_inpatient',
             'number_diagnoses']

from sklearn import preprocessing
my_scaler = preprocessing.StandardScaler().fit(dataframe1[scale_attr])
scale_data = my_scaler.transform(dataframe1[scale_attr])

scale_dataframe = pd.DataFrame(data = scale_data, columns = scale_attr, index = dataframe1.index )
dataframe1.drop(scale_attr, axis = 1, inplace = True)
dataframe1 = pd.concat([dataframe1, scale_dataframe], axis = 1)

#dataframe1['number_outpatient']


# In[11]:


#converting categorical attributes to numerical i.e., binary attributes
cat_attributes = ['race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id',
                  'max_glu_serum','A1Cresult',
                  'metformin','glimepiride','glipizide','glyburide','pioglitazone',
                  'rosiglitazone','insulin']

for i in cat_attributes: 
    dataframe1 = pd.get_dummies(dataframe1, prefix = [i], columns = [i])  


# In[12]:


dataframe1.to_csv('KeerProcessData1.csv')


# In[13]:


dataframe1.shape


# In[14]:


#create concise table to understand the data

data_type = dataframe1.dtypes.values
missing_count = dataframe1.isnull().sum().values
#[]
#for att in dataframe1.columns:
#    dataframe1.isnull().sum().values

unique_count = []
for attr in dataframe1.columns:
    unique_count.append(dataframe1[attr].unique().shape[0])
    
info ={'Attributes': dataframe1.columns,
       'Attribute_Type': data_type,
       'MissingValue_Count':missing_count,
       'UniqueValue_Count': unique_count
       }
col_names = {'Attributes','Attribute_Type','MissingValue_Count','UniqueValue_Count'}

info_tab = pd.DataFrame(info,columns = col_names)
info_tab


# In[ ]:


#above table shows that the data now has numeric data only with float and int format 


# In[15]:


#create x and y

x = dataframe1.drop(['readmitted'], axis = 1)
y = dataframe1['readmitted']


# In[ ]:


#y


# In[16]:


#split x and y into train test split
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)


# In[17]:


print x_train.shape
print y_train.shape
print x_test.shape
print y_test.shape


# In[ ]:


#x_train.to_csv('TrainX.csv')
#y_train.to_csv('TrainY.csv')
#x_test.to_csv('TestX.csv')
#y_test.to_csv('TestY.csv')


# In[ ]:


#PCA
#from sklearn.decomposition import TruncatedSVD
#from sklearn.decomposition import PCA

#dr = TruncatedSVD()
#dr=PCA()
#dr.fit(x_train)
#dr.fit(x_test)
#x_train=dr.transform(x_train)
#x_test=dr.transform(x_test)


# In[18]:


from sklearn.neural_network import MLPClassifier


mlp = MLPClassifier()
model = mlp.fit(x_train,y_train) 
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score 
MLPAccuracy = accuracy_score(y_test, pred) 
print("Neural Network Accuracy:",MLPAccuracy)

from sklearn.metrics import f1_score 
F1score = f1_score(y_test, pred, average= 'micro')
print("F1Score:",F1score)


# In[19]:


#Perceptron
from sklearn.linear_model import Perceptron

per = Perceptron()
model = per.fit(x_train,y_train)
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
PerAccuracy = accuracy_score(y_test, pred)
print("Perceptron Accuracy:",PerAccuracy)

from sklearn.metrics import f1_score
F1score = f1_score(y_test, pred, average= 'micro') 
print("F1Score:",F1score)

#from sklearn.metrics import confusion_matrix 
#print(confusion_matrix(y_test, pred))


# In[20]:


#LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model=lr.fit(x_train,y_train)
pred_nb = model.predict(x_test)

from sklearn.metrics import accuracy_score
LRAccuracy = accuracy_score(y_test, pred)
print("LR Accuracy:",LRAccuracy)

from sklearn.metrics import f1_score
F1score = f1_score(y_test, pred, average= 'micro') 
print("F1Score:",F1score)

#from sklearn.metrics import confusion_matrix 
#print(confusion_matrix(y_test, pred_nb))


# In[ ]:


#Gaussian NB
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
model = nb.fit(x_train,y_train)
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
GNBAccuracy = accuracy_score(y_test, pred)
print("Naive base Accuracy:",GNBAccuracy)

from sklearn.metrics import f1_score
F1score = f1_score(y_test, pred, average= 'micro') 
print("F1Score:",F1score)

#from sklearn.metrics import confusion_matrix 
#print(confusion_matrix(y_test, pred_nb))


# In[ ]:


#Decision Tree
from sklearn import tree
dt = tree.DecisionTreeClassifier()
model=dt.fit(x_train,y_train)
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
DTAccuracy = accuracy_score(y_test, pred)
print("Decision Tree Accuracy:",DTAccuracy)

from sklearn.metrics import f1_score
F1score = f1_score(y_test, pred, average= 'micro') 
print("F1Score:",F1score)


# In[ ]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
model = rf.fit(x_train,y_train)
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
RFAccuracy = accuracy_score(y_test, pred)
print("Random Forest Accuracy:",RFAccuracy)

from sklearn.metrics import f1_score
F1score = f1_score(y_test, pred, average= 'micro') 
print("F1Score:",F1score)

#from sklearn.metrics import confusion_matrix 
#print(confusion_matrix(y_test, pred))



# In[ ]:


#Bagging
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier()
model = bag.fit(x_train,y_train)
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
BagAccuracy = accuracy_score(y_test, pred)
print("Random Forest Accuracy:",BagAccuracy)

from sklearn.metrics import f1_score
F1score = f1_score(y_test, pred, average= 'micro') 
print("F1Score:",F1score)


# In[ ]:


#ExtraTrees
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier()
model = et.fit(x_train,y_train)
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
ETAccuracy = accuracy_score(y_test, pred)
print("Extra Trees Accuracy:",ETAccuracy)

from sklearn.metrics import f1_score
F1score = f1_score(y_test, pred, average= 'micro') 
print("F1Score:",F1score)


# In[ ]:


#plot and compare scores 

x_axis = np.arange(9)
y_axis = [MLPAccuracy , LRAccuracy,
          GNBAccuracy, DTAccuracy, RFAccuracy,
          BagAccuracy,  PerAccuracy, ETAccuracy]

pt.bar(x_axis, y_axis, width=0.5)
pt.xticks(x_axis+0.5/10.,('MLP','LR',
           'GNB','DT','RFt',
            'Bag','Per','ET'))
pt.ylabel('F1 Score')

pt.show()


# In[ ]:


#since MLP and Adaboost gave quite good values we are validating both


# In[ ]:


#Cross_validation
from sklearn.cross_validation import cross_val_score
MLP_score = cross_val_score(mlp, x_train, y_train, cv=5).mean()
print('MLP cross validation score', MLP_score)
AdaB_score = cross_val_score(adbo, x_train, y_train, cv=5).mean()
print('Adaboost cross validation score', AdaB_score)


# In[ ]:


#Parameter turning using GridSerachCV

from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import GridSearchCV

parameters ={"solver":['sgd'],
            "hidden_layer_sizes":(15,),
            "random_state":[1]}
            
mlp = MLPClassifier()
grid = GridSearchCV(mlp,parameters,cv=5,scoring='accuracy')
grid.fit(x_train, y_train)
print(grid.best_score_)
#print(grid.best_estimators_.)


# In[ ]:


from sklearn.neural_network import MLPClassifier


mlp = MLPClassifier(solver='sgd',hidden_layer_sizes=15,random_state=1)
model = mlp.fit(x_train,y_train) 
pred = model.predict(x_test)

from sklearn.metrics import accuracy_score 
MLPAccuracy = accuracy_score(y_test, pred) 
print("Neural Network Accuracy:",MLPAccuracy)

from sklearn.metrics import f1_score 
F1score = f1_score(y_test, pred, average= 'micro')
print("F1Score:",F1score)

