
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('smsspamcollection.tsv', sep = '\t')

# cheking data
# 1. if any missing data?
data.isnull().sum()
# 2. length of data frame
len(data)
# 3. sample categories
data['label'].unique()
data['label'].value_counts()

## machine learning model
X = data[['length', 'punct']]
Y = data['label']

x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size = 0.3, random_state = 42)

## Model 1: Logistic Regression
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(solver = 'lbfgs')
model1.fit(x_train, y_train)
pr1 = model1.predict(x_test)

result1 = metrics.confusion_matrix(y_test, pr1)
print(result1)

df1 = pd.DataFrame(metrics.confusion_matrix(y_test, pr1), 
                    index = ['ham', 'spam'],
                    columns = ['ham', 'spam'],
                    )

print(metrics.classification_report(y_test, pr1))
print(metrics.accuracy_score(y_test, pr1))

## Model 2: naive Bayesian
from sklearn.naive_bayes import MultinomialNB

model2 = MultinomialNB()
model2.fit(x_train, y_train)
pr2 = model2.predict(x_test)

df2 = pd.DataFrame(metrics.confusion_matrix(y_test, pr2), 
                    index = ['ham', 'spam'],
                    columns = ['ham', 'spam'],
                    )

print(df2)
print(metrics.classification_report(y_test, pr2))
print(metrics.accuracy_score(y_test, pr2))

## model 3: support vector machine
from sklearn.svm import SVC

model3 = SVC(gamma = 'auto')
model3.fit(x_train, y_train)
pr3 = model3.predict(x_test)

df3 = pd.DataFrame(metrics.confusion_matrix(y_test, pr3), 
                   index = ['ham', 'spam'],
                   columns = ['ham', 'spam'],
                   )

print(df3)
print(metrics.classification_report(y_test, pr3))
print(metrics.accuracy_score(y_test, pr3))

