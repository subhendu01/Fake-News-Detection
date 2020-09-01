import pickle
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Read the CSV data file
df=pd.read_csv('news.csv')

# Column names
# print(df.columns)
# data
# print(df.head(2).to_string())

# Get the labels
labels=df.label
# labels.head()
X = df['text']
y = df['label']

# Split the dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=7)

# Initialize a TfidfVectorizer and filter stop word before processing the data
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize a PassiveAggressiveClassifier
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(tfidf_train, y_train)
# Predict on the test set and calculate accuracy
y_pred = classifier.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')

#Checking the performance of our model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))

# Build confusion matrix and plotting it
labels = ['FAKE', 'REAL']
cm = confusion_matrix(y_test, y_pred, labels )
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# # Serializing the file
with open('model/model2.pickle', 'wb') as handle:
    pickle.dump(classifier, handle, protocol= pickle.HIGHEST_PROTOCOL)
