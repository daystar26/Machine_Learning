""""
Name : Darryl Ramgoolam 

Exercise: First Machine Learning application, 
Utilizes the load_Iris data sit. 

Machine Learning With Python

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn                 # Library that came with Book

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load Iris Data
iris_dataset = load_iris()

# Print the keys of the Iris Data
print('Keys of Iris_dataset: \n{}\n'.format(iris_dataset.keys()))

# Print the Target Names
print('Target Names : {}\n'.format(iris_dataset['target_names']))

# Print the Features 
print ('Features : \n {}\n'.format(iris_dataset['feature_names']))

# Print the shiris_datasetape of the data
print ('Shame of data: \n {}\n'.format(iris_dataset['data'].shape))

# Use the train_test_split() to split the data into 75%/25%
# where 75% of the data set will be used to train the model
# and 25% of the data set will be used to test the model
X_train, x_test, Y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# Verify the split worked correctly
print('X_train  shape:{}'.format(X_train.shape))
print('x_test  shape:{}'.format(x_test.shape))
print('Y_train  shape:{}'.format(Y_train.shape))
print('y_test  shape:{}'.format(y_test.shape))
'''
# Create Plots from Iris Data Set
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset['feature_names'])
graph = pd.scatter_matrix(iris_dataframe, c = Y_train, figsize=(15,15), marker='o',
                          hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

# Show plots
plt.show()
'''
# Set up for k-nearest neighbors classification algorithm
knn = KNeighborsClassifier(n_neighbors=1)

# Build model using traning set
# X_Train is the traning data 
# Y_Train is the traning labels
knn.fit(X_train, Y_train)

# Make a Prediciton using made up data
X_New = np.array([[5, 2.7, 1, 0.2]])    # Needs to be 2D array
new_prediction = knn.predict(X_New)

print('\nPrediction : {}'.format(new_prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][new_prediction]))

# Make Prediciton using test data
y_pred = knn.predict(x_test)

# Calculate the test accuracy
print('Test Set score is: {:.2f}'.format(np.mean(y_pred == y_test)))