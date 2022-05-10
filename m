Linear Regression

#import packages
import matplotlib.pyplot as plt
import pandas as pd

# Read Dataset
dataset=pd.read_csv('/content/sample_data/hours.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

# print(X, y)
dataset.head()
# dataset.shape
dataset

# Import the Linear Regression and Create object of it
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
Accuracy=regressor.score(X, y)*100
print(Accuracy)

# Predict the value using Regressor Object
y_pred=regressor.predict([[10]])
print(y_pred)

# Take user input
hours=int(input('Enter the no of hours'))

#calculate the value of y
eq=regressor.coef_*hours+regressor.intercept_
eq[0]
regressor.predict(X)
plt.plot(X,y,'o')
plt.plot(X,regressor.predict(X));
plt.show()




#DT

import pandas as pd
import numpy as np

#reading Dataset
dataset=pd.read_csv('/content/sample_data/tree1.csv')
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,5]

dataset.head()
X.head()
y.head()

#Perform Label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X=X.apply(le.fit_transform)
X
y[8]

#import Decesion Tree Classifier 
from sklearn.tree import DecisionTreeClassifier
# Create decision tree classifer object
regressor=DecisionTreeClassifier()
# Train model
regressor.fit(X.iloc[:,1:5],y)

x_in=np.array([1,1,0,0])
y_pred=regressor.predict([x_in])
print(y_pred)


# from sklearn.externals.six import StringIO
# from IPython.display import Image
# from sklearn.tree import export_graphviz
# import pydotplus

# dot_data=StringIO()
# export_graphviz(regressor,out_file=dot_data,filled=True,rounded=True,special_characters=True)

# #Draw Graph
# graph=pydotplus.graph_from_dot_data(dot_data.getvalue())

# # Show graph & Create png File
# graph.write_png("tree.png")


#KNN
#import the packages
import pandas as pd
import numpy as np

#Read dataset
dataset=pd.read_csv('/content/sample_data/kdata.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,2].values

dataset

X, y

#import KNeighborshood Classifier and create object of it
from sklearn.neighbors import KNeighborsClassifier

#Creating model
classifier=KNeighborsClassifier(n_neighbors=3)
# Training model
classifier.fit(X,y)


#predict the class for the point(6,6)
X_test=np.array([6,6])
# Predictions for test data
y_pred=classifier.predict([X_test])
print(y_pred)


# KNeighborsClassifier looks for the 5 nearest neighbors
#If set to uniform, all points in each neighbourhood have 
#equal influence in predicting class i.e. predicted class is the class with highest number of points in the neighbourhood.
classifier=KNeighborsClassifier(n_neighbors=3,weights='distance')
classifier.fit(X,y)

#predict the class for the point(6,6)
X_test=np.array([6,2])
y_pred=classifier.predict([X_test])
print(y_pred)

#kmean

#import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read dataset
dataset=pd.read_csv("/content/sample_data/dataset.csv")
f1=dataset['X'].values
f2=dataset['Y'].values
X= np.array(list(zip(f1,f2)))
X

#centroid points
C_x=np.array([0.1,0.3])
C_y=np.array([0.6,0.2])
centroids=C_x,C_y

#plot the given points
colmap = {1:'r', 2: 'b'}
plt.scatter(f1, f2, color='k')
plt.show()

#for i in centroids():
plt.scatter(C_x[0],C_y[0], color=colmap[1])

plt.scatter(C_x[1],C_y[1], color=colmap[2])
plt.show()

C = np.array(list((C_x, C_y)), dtype=np.float32)
print (C)

#plot given elements with centroid elements
plt.scatter(f1, f2, c='#050505')
plt.scatter(C_x[0], C_y[0], marker='*', s=200, c='r')
plt.scatter(C_x[1], C_y[1], marker='*', s=200, c='b')
plt.show()


#plot the given points
plt.scatter(f1, f2, c='#050505', s=7)
plt.show()
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
plt.show()

#import KMeans class and create object of it
from sklearn.cluster import KMeans
model=KMeans(n_clusters=2,random_state=0)
model.fit(X)
labels=model.labels_
print(labels)

#using labels find population around centroid
count=0
for i in range(len(labels)):
  if (labels[i]==1):
    count=count+1
    
print('No of population around cluster 2:',count-1)


#Find new centroids
new_centroids = model.cluster_centers_

print('Previous value of m1 and m2 is:')
print('M1==',centroids[0])
print('M1==',centroids[1])

print('updated value of m1 and m2 is')
print('M1==',new_centroids[0])
print('M1==',new_centroids[1])
