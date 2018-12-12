
# coding: utf-8

# In[36]:


# This version one hot encodes the drinks, drugs, smokes responses
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC


# In[37]:


profiles=pd.read_csv("profiles.csv")
print(profiles.shape)

#profiles.head(3)
print(profiles.columns)


# In[38]:


# get dummies creates new dataframe
smokes=pd.get_dummies(profiles['smokes'].fillna('no answer'),prefix='smokes')


# In[39]:


# use pandas get_dummies to one hot encode drinks,drugs and job , prefix identifies how the prefix of the new columns

drinks=pd.get_dummies(profiles['drinks'].fillna('no answer'),prefix='drinks')

drugs=pd.get_dummies(profiles['drugs'].fillna('no answer'),prefix='drugs')
job=pd.get_dummies(profiles['job'].fillna('no answer'),prefix='job')
#
# pd.concat concatenates dataframes, axis=1 says to add columns
#
profiles=pd.concat([profiles,smokes,drinks,drugs],axis=1)


# In[40]:


# map orientation repsonses to codes
orientation_mapping={"straight":0,"gay":1,"bisexual":2}
profiles["orientation_code"]=profiles.orientation.map(orientation_mapping)


# In[41]:


#plt.hist(profiles['orientation_code'])
plt.xlabel("Orientation")
plt.ylabel("Frequency")

N, bins, patches = plt.hist(profiles['orientation_code'], 3, ec="k")

cmap = plt.get_cmap('jet')
straight = cmap(0.5)
gay =cmap(0.25)
bisexual = cmap(0.8)


for i in range(0,1):
    patches[i].set_facecolor(straight)
for i in range(1,2):
    patches[i].set_facecolor(gay)
for i in range(2,3):
    patches[i].set_facecolor(bisexual)

#create legend
handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [straight,gay,bisexual]]
labels= ['straight','gay','bisexual']
plt.legend(handles, labels)
plt.title('Orientation')
plt.show()


# In[42]:


smoke_mapping={'no':1,'sometimes':2,'when drinking':3,'yes':4,'trying to quit':5}
profiles['smokes_code']=profiles.smokes.map(smoke_mapping)

#plt.hist(profiles['smokes_code'])
plt.xlabel("Smokes")
plt.ylabel("Frequency")

N, bins, patches = plt.hist(profiles['smokes_code'], 5, ec="k")

cmap = plt.get_cmap('jet')
no = cmap(0.15)
sometimes =cmap(0.3)
when_drinking = cmap(0.45)
yes=cmap(.6)
trying_to_quit=cmap(.85)



patches[0].set_facecolor(no)
patches[1].set_facecolor(sometimes)
patches[2].set_facecolor(when_drinking)
patches[3].set_facecolor(yes)
patches[4].set_facecolor(trying_to_quit)

#create legend
handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [no,sometimes,when_drinking,yes,trying_to_quit]]
labels= ['no','sometimes','when_drinking','yes','trying_to_quit']
plt.legend(handles, labels)
plt.title('Smokes')
plt.show()


# In[43]:


# combine essay responses into 1 column and remove NANs, create column containing combined essays length
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

# Removing the NaNs
all_essays =profiles[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
profiles['all_essays']=all_essays.str.lower()

profiles["essay_len"] = all_essays.apply(lambda x: len(x))


# In[44]:


# define function to calculate average number of words in combined essays
# add new column of average word lengths
def avg_word_len(essay):
    words_in_essay=essay.split(' ')
    num_words=len(words_in_essay)
    return len(essay)/num_words

profiles["avg_word_len"]=[avg_word_len(all_essays[x]) for x in range(len(all_essays)) ]


# In[45]:


# caclulate number of times 'I' and 'me' are used in essay
profiles["i_count"]=all_essays.apply(lambda x: x.count(" i "))
profiles["me_count"]=all_essays.apply(lambda x: x.count(" me "))
profiles["i_or_me"]=profiles["i_count"]+profiles["me_count"]
profiles['you_count']=all_essays.apply(lambda x: x.count(' you '))


# In[46]:


# find all the one_hot encoded columns created by pd.get_dummies by iterating over the columns and matching on prefix
smoke_cols = [col for col in profiles.columns if 'smokes_' in col]
drink_cols= [col for col in profiles.columns if 'drinks_' in col]
drug_cols=[col for col in profiles.columns if 'drugs_' in col]


# In[47]:


# select features
# select the numeric features
feature_data = profiles[ ['essay_len', 'avg_word_len','you_count','orientation_code']]


# standardize features
x = feature_data.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
feature_data.fillna(value=0,inplace=True)  # remove Nan's from drugs_code I could see

# add the one_hot_encoded columns to feature data
feature_data =pd.concat([feature_data,smokes,drinks,drugs],axis=1)
feature_data.head()
# select target feature as y
y=feature_data["orientation_code"]


# In[48]:


# drop orientation_code from the feature dataframe since it's the target
X=feature_data.drop('orientation_code',axis=1)
X.head()


# In[49]:


# split training and test data 80% training, 20% test
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state=42)


# In[50]:


print(X.shape)
print(y.shape)
print(X_train.shape,X_test.shape,y_test.shape)
X.head()


# In[51]:


regressor = KNeighborsRegressor(n_neighbors=3)
regressor.fit(X_train,y_train)


# In[52]:


guess=regressor.predict(X_test)
score=regressor.score(X_train,y_train)
print('score=',score)


# In[53]:


# calculate simple accuracy
j=0
correct=0
incorrect=0
for key,value in y_test.iteritems():
    #print(key,value)
    if  guess[j]==value:
        correct+=1
        
    else:
        incorrect+=1
    j+=1
print ('j=',len(guess),' correct=',correct,' incorrect=',incorrect,'accuracy=',correct/len(y_test))


# In[54]:


mlr=LinearRegression()
mlr.fit(X_train,y_train)
print('training score=',mlr.score(X_train,y_train))


# In[55]:


print('test score=',mlr.score(X_test,y_test))


# In[56]:


guess=print(mlr.predict(X_test))


# In[57]:


classifier=KNeighborsClassifier(n_neighbors=3)
y=profiles['orientation_code']
# split training and test data 80% training, 20% test
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state=42)


# In[58]:


classifier.fit(X_train,y_train)


# In[59]:


guess=classifier.predict(X_test)


# In[60]:


correct=0
incorrect=0
j=0
for key,value in  y_test.iteritems():
    if guess[j]==value:
        correct+=1
    else:
        incorrect+=1
    j+=1
print('length of guess=',len(guess),'correct=',correct,' incorrect=',incorrect)
print(classifier.score(X_test,y_test))


# In[61]:


# change weights
classifier=KNeighborsClassifier(n_neighbors=3,weights='distance')
y=profiles['orientation_code']
# split training and test data 80% training, 20% test
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state=42)
classifier.fit(X_train,y_train)
guess=classifier.predict(X_test)
correct=0
incorrect=0
j=0
for key,value in  y_test.iteritems():
    if guess[j]==value:
        correct+=1
    else:
        incorrect+=1
    j+=1
print('length of guess=',len(guess),'correct=',correct,' incorrect=',incorrect)
print(classifier.score(X_test,y_test))


# In[62]:


svm=SVC(kernel='poly',gamma='auto')
svm.fit(X_train,y_train)


# In[63]:


svm_guess=svm.predict(X_test)


# In[64]:


correct=0
incorrect=0
j=0
for key,value in  y_test.iteritems():
    if svm_guess[j]==value:
        correct+=1
    else:
        incorrect+=1
    j+=1
print('length of guess=',len(svm_guess),'correct=',correct,' incorrect=',incorrect)
print("SVM polynomial score=",svm.score(X_test,y_test))


# In[66]:


svm=SVC(kernel='rbf',gamma='auto')
svm.fit(X_train,y_train)
svm_guess=svm.predict(X_test)
correct=0
incorrect=0
j=0
for key,value in  y_test.iteritems():
    if svm_guess[j]==value:
        correct+=1
    else:
        incorrect+=1
    j+=1
print('length of guess=',len(svm_guess),'correct=',correct,' incorrect=',incorrect)
print("SVM rbf score=",svm.score(X_test,y_test))

