#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data into a Pandas dataframe
df = pd.read_csv('creditcard.csv')

X = df.drop("Class", axis=1)
y = df["Class"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocess the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Train a random forest classifier
model2 = RandomForestClassifier(random_state=42)
model2.fit(X_train, y_train)


# Evaluate the model on the testing set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)

# Evaluate the model on the testing set
y_pred2 = model2.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)
precision2 = precision_score(y_test, y_pred2)
recall2 = recall_score(y_test, y_pred2)
f1_2 = f1_score(y_test, y_pred2)

print('Accuracy2:', accuracy2)
print('Precision2:', precision2)
print('Recall2:', recall2)
print('F1 score2:', f1_2)

# Create a DataFrame to store the predictions and actual values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)
results2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2}, index=y_test.index)

# Display the first 10 rows of the results DataFrame
# print(results.head(10))

#VISUALIZATION
# Get the number of fraudulent and non-fraudulent cases
fraud_count = len(df[df['Class'] == 1])
non_fraud_count = len(df[df['Class'] == 0])

# Create a bar plot
plt.bar(['Fraudulent', 'Non-Fraudulent'], [fraud_count, non_fraud_count])
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Distribution of Target Classes')

# Add text labels to the bars
for i, count in enumerate([fraud_count, non_fraud_count]):
    plt.text(i, count + 100, str(count), ha='center')

plt.show()

