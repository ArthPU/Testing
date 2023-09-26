#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


xpoints = np.array([0, 6])
ypoints = np.array([0, 250])

plt.plot(xpoints, ypoints)
plt.show()


# In[5]:


x = np.random.normal(170, 10, 250)

plt.hist(x)
plt.show() 


# In[3]:


#Dot Diagram
x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x,y, color='indigo',s=50, marker='x')
plt.title("DOT DIAGRAM")
plt.xlabel("Age")
plt.ylabel("Money")
plt.show()


# In[4]:


#Boxplot Diagram

# Generate random data
np.random.seed(100)
data = np.random.randint(50)

#Create a boxplot
figure,axis = plt.subplots()
plt.boxplot(data)

# Add labels
axis.set_title("Boxplot")
axis.set_xlabel("X")
axis.set_ylabel("Y")

plt.show()


# In[5]:


x = ['A','B','C','D']
y = [10,20,10,50]

plt.bar(x,y)
plt.title("Bar Diagram")
plt.xlabel("Name")
plt.ylabel("Age")

plt.show()


# In[6]:


# Stacked Bar Diagram 

x = ['A','B','C','D']
y1 = [10,21,12,30]
y2 = [15,32,12,23]

plt.bar(x,y1)
plt.bar(x,y2, bottom=y1)

plt.title("Stacked Bar Diagram")
plt.xlabel("Name")
plt.ylabel("Age")

plt.show()


# In[7]:


#Area Chart

x = np.arange(0,10,0.3)
y1 = np.sin(x)
y2 = np.cos(x)

plt.fill_between(x,y1,y2, color = 'magenta')
plt.title("Area Chart")
plt.xlabel("Name")
plt.ylabel("Age")

plt.show()


# In[8]:


x = np.array([0,1,2,3])
y = np.array([3,8,1,10])

#Graph 1
plt.subplot(2,1,1)
plt.plot(x,y, linestyle='-.', linewidth=5, color='blue', label='Solid Label')

#Graph 2
x = np.array([0,1,2,3])
y = np.array([10,15,4,21])
plt.subplot(2,1,2)
plt.plot(x,y)

plt.show()


# In[ ]:





# In[16]:


#Plot 1 
x = np.array([1,4,2,5])
y = np.array([7,6,3,9])


plt.subplot(2,1,1)
plt.plot(x,y,linestyle='-.', linewidth=5, color='blue', label='Solid Label')

#PLot 2 
x = np.array([1,4,2,5])
y = np.array([7,6,3,9])

plt.subplot(2,1,2)
plt.plot(x,y)

plt.show()


# In[17]:


x = np.array([1,4,2,5])
y = np.array([7,6,3,9])


plt.subplot(1,2,1)
plt.plot(x,y)
plt.show()


# In[19]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

url = 'https://query.data.world/s/m223x7c6twqamqhrnzjewz4sxqic7m?dws=00000'
data = pd.read_csv(url)
data.head()



x = data["Selected Cause of Death"]
y = data["Deaths"]
plt.bar(x,y)
plt.title("Bar Data")
plt.xticks(rotation=45)
# plt.xlabel("Cause of Death")
plt.ylabel("Death")

plt.show()


# In[17]:


x = [10,20,15,30]
y = [25,16,30,5]

plt.bar(x,y)
plt.title("Bar")
plt.xlabel("Death")
plt.ylabel("Men")
plt.show()


# In[23]:


sorted_data = data.sort_values(by='Deaths', ascending=False)

plt.figure(figsize=(10,6))
plt.bar(sorted_data["Selected Cause of Death"], sorted_data["Deaths"])
plt.title("Sorted Data")
plt.xlabel("Selected Cause")
plt.ylabel("Death")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()


# In[24]:


sorted_data = data.sort_values(by='Deaths', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Deaths', y='Selected Cause of Death', data=sorted_data)
# plt.bar(sorted_data["Selected Cause of Death"], sorted_data["Deaths"])
plt.title("Sorted Data")
plt.xlabel("Selected Cause")
plt.ylabel("Death")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()


# In[18]:


ds = pd.read_csv('dm_office_sales.csv')
ds1 = ds.head()
ds1


# In[20]:


ds = pd.read_csv('dm_office_sales.csv')
head = ds.head()


#Seaborn
sc =sns.scatterplot(x="salary", y ="sales", data=ds)



# In[28]:


ds = pd.read_csv('dm_office_sales.csv')
head = ds.head()


#Seaborn
plt.figure(figsize = (12,4), dpi=400)
sc =sns.scatterplot(x="salary", y ="sales",hue="level of education", data=ds)



