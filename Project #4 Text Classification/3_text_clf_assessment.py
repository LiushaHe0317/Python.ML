
#1: Perform imports and load the dataset into a pandas DataFrame
import numpy as np
import pandas as pd

data = pd.read_csv('moviereviews2.tsv', sep = '\t')

#2: Check for missing values:
print(data.isnull().sum())
print(len(data))

#3: Remove NaN values
# step #1
data.dropna(inplace = True)
print(data.isnull().sum())
print(len(data))

# step #2
blanks = []
for i, lb, rv in data.itertuples():
    print(lb)
    if rv.isspace():
        blanks.append(i)
data.drop(blanks, inplace = True)
print(data.isnull().sum())
print(len(data))

#4: Take a quick look at the label column
data['label'].value_counts()

#5: Split the data into train & test sets
from sklearn.model_selection import train_test_split

X = data['review']
Y = data['label']

x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size = .3, random_state = 42)

#6: Build a pipeline to vectorize the data, 
#   then train and fit a model
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

text_clf = Pipeline([('TF-IDF', TfidfVectorizer()),('clf', LinearSVC())])

text_clf.fit(x_train, y_train)

#7: Run predictions and analyze the results
pr = text_clf.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(y_test, pr))
print(classification_report(y_test, pr))
print(accuracy_score(y_test, pr))


