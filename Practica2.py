#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
col_names = ['Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure','Serum Cholestoral','Fasting Blood Sugar','Resting Electrocardiographic Results','Maximum Heart Rate Achieved','Exercise Induced Angina','ST Depression ','Slope of the Peak Exercise','Number of Major Vesselslabel','Thal','label']
# load dataset
pima = pd.read_csv("heart.csv", header=None, names=col_names)



feature_cols = ['Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure','Serum Cholestoral','Fasting Blood Sugar','Resting Electrocardiographic Results','Maximum Heart Rate Achieved','Exercise Induced Angina','ST Depression ','Slope of the Peak Exercise','Number of Major Vesselslabel','Thal']
X = pima[feature_cols] # Features
y = pima.label # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image 
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import pydotplus

pima.head()

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Heart Disease UCI.png')
Image(graph.create_png())

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Heart Disease UCI 1.png')
Image(graph.create_png())


# In[ ]:





# In[ ]:




