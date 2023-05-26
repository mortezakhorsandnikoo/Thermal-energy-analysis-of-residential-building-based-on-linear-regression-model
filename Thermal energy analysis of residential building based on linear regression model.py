#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries
import numpy as np
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sb
data=pd.read_csv(r"D:/AI_Machinlearning/datasets/ENERGY/data_energy.csv")
data.head(10)


# In[3]:


data.describe()


# In[5]:


#correlation
d=data.drop([ "X0"] , axis= 1)

corr_matrix=d.corr()
corr_matrix


# In[6]:


mask=np.zeros_like(corr_matrix , dtype=bool)
mask[np.triu_indices_from(mask)] = True

print(mask.shape)
print(corr_matrix.corr().shape)


# In[7]:


#correlation graph
plt.figure(figsize=(10,10) , dpi=300)

sb.heatmap(corr_matrix , vmin=-1 ,cmap='coolwarm', annot=True , robust = True , cbar= True, mask=mask,
          cbar_kws={"shrink":0.8}, annot_kws={"size":12} ,  linewidth=0.5)


# In[12]:


plt.figure(figsize=(2,2) , dpi=300 )


g = sb.pairplot(d, diag_kind="auto", corner= True  ,aspect=1 , height=1.5, diag_kws=dict(fill=False))
g.map_lower(sb.kdeplot, levels=4, color=".2" )


# In[19]:


X= np.array(data.drop(["Y" , "X0"] , axis= 1))
y=np.array(data["Y"])



# In[36]:


#train and accuracy
acctrain=[]
acctest=[]
for i in range(1,21):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state = i)
    if i==1:
        print('X_train dimension= ', X_train.shape)
        print('X_test dimension= ', X_test.shape)
        print('y_train dimension= ', y_train.shape)
        print('y_test dimension= ', y_test.shape)
        print("----------------------------------")
        theta= np.linalg.pinv(X_train)@y_train
    y_hat= X_train@theta
    numerator=sum ((y_train - y_hat)**2)
    Denominator=sum ((y_train - y_train.mean())**2)
    rse= numerator / Denominator
    R_squared= 1 - rse
    acctrain.append(round(float(R_squared*100), 2))
    y_hattest= X_test@theta
    numerator2=sum ((y_test - y_hattest)**2)
    Denominator2=sum ((y_test - y_test.mean())**2)
    rsetest= numerator2 / Denominator2
    R_squared2= 1 - rsetest
    acctest.append(round(float(R_squared2*100), 2))




print(acctrain)
print(acctest)
print("----------------------------------")
print(np.array(acctrain).mean())
print(np.array(acctest).mean())


# In[21]:


np.set_printoptions(precision=2)


# In[22]:


fig, ax = plt.subplots(figsize=(12,6) , dpi=300)
#fig.suptitle('test title', fontsize=120)

font1 = {'family':'serif','color':'darkred','size':12}
font2 = {'family':'serif','color':'darkred','size':15}

x4= [i for i in range (1,21)] 

plt.xticks(np.arange(min(x4), max(x4)+1, 1.0))


ax.set_xlabel('Epochs', fontsize = 15,fontdict = font2)
ax.set_ylabel('Accuracy', fontsize =15, fontdict = font2)


#plt.legend(["Train" , "Test"], loc="upper left" , fontsize="40")


plt.scatter(x4 , acctrain, alpha=0.6 , c="red",s = 100)
plt.scatter(x4 , acctest, alpha=0.6 , c="black" , s=100)
plt.plot(x4 , acctrain, alpha=0.6 , color="red", label='Train')
plt.plot(x4 , acctest, alpha=0.6 , color="black", label='Test')
plt.legend()

plt.show()


# In[23]:


import pandas as pd

d={"train_accuracy":acctrain   , "test_accuracy": acctest  }

dataframe=pd.DataFrame(d , index=range(1,21))
display(dataframe)


# In[30]:


fig = plt.figure(figsize=(8, 8) , dpi=300)
ax = fig.add_subplot(111)

x1=np.linspace(0,30,num=130)

plt.scatter(x1 , y_test, alpha=0.6 , c="black" , marker="x" , s=50  , label='y_real')
plt.scatter(x1 , y_hattest, alpha=0.6 , c="red" , s=50,label= "y_predicted" )
plt.legend()
plt.title("The Difference Between y_real and y_predicted" , font1)
#plt.scatter(y_hattest, y_test , c="red" , alpha=0.6)
plt.plot()
plt.show()

