
import numpy as np
import pandas as pd

data = pd.read_csv('moviereviews.tsv', sep = '\t')

data.head()
len(data)
data.isnull().sum()

## remove missing reviews
data.dropna(inplace = True)
data.isnull().sum()
len(data)
# removed 35 reviews
## remove empty string data
blanks = []

for i, lb, rv in data.itertuples():
    if rv.isspace():
        blanks.append(i)
data.drop(blanks, inplace = True)
print(len(data))
data.isnull().sum()

from sklearn.model_selection import train_test_split

X = data['review']
Y = data['label']

x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size = 0.3, random_state = 42)

## Vectorization
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

text_clf = Pipeline([('TF-IDF',TfidfVectorizer()),
                     ('clf', LinearSVC())])
    
text_clf.fit(x_train, y_train)
pr = text_clf.predict(x_test)

## evaluate the model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

result1 = confusion_matrix(y_test, pr)
result2 = classification_report(y_test, pr)
result3 = accuracy_score(y_test, pr)
