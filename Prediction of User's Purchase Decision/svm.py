#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machine (SVM)

# ## Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# ## Splitting the dataset into the Training set and Test set


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# ## Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Training the SVM model on the Training set


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


# ## Predicting a new result

print(classifier.predict(sc.transform([[30,87000]])))


# ## Predicting the Test set results


y_pred = classifier.predict(X_test)

# ## Making the Confusion Matrix


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# Training set visualization
plt.figure(figsize=(8, 6))
X_set, y_set = sc.inverse_transform(X_train), y_train
plt.scatter(X_set[y_set == 0, 0], X_set[y_set == 0, 1], color='salmon', label='Not Purchased')
plt.scatter(X_set[y_set == 1, 0], X_set[y_set == 1, 1], color='dodgerblue', label='Purchased')
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Test set visualization
plt.figure(figsize=(8, 6))
X_set, y_set = sc.inverse_transform(X_test), y_test
plt.scatter(X_set[y_set == 0, 0], X_set[y_set == 0, 1], color='salmon', label='Not Purchased')
plt.scatter(X_set[y_set == 1, 0], X_set[y_set == 1, 1], color='dodgerblue', label='Purchased')
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

