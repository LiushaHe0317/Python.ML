
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

data = pd.read_csv('smsspamcollection.tsv', sep = '\t')

# split data
X = data['message']
Y = data['label']
x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size = 0.3, random_state = 42)

## Count Vectorizing the raw text data
vect1 = CountVectorizer()
# APROAH 1: Fit the vectorizer to the data
# vect1.fit(x_train)
# x_train_counts = vect1.transform(x_train)
# APPROACH 2: TRANSFORM THE ORIGINAL MESSEGE --> VECTOR
x_train_counts = vect1.fit_transform(x_train)

## TF-IDF
transformer = TfidfTransformer()
x_train_tfidf = transformer.fit_transform(x_train_counts)

# Vectorization
vect2 = TfidfVectorizer()
x_train_tfidf = vect2.fit_transform(x_train)

## import classifier
model1 = LinearSVC()
model1.fit(x_train_tfidf, y_train)

## Create a Pipeline object
test_clf = Pipeline([('tf-idf', TfidfVectorizer()),('model1', LinearSVC())])
test_clf.fit(x_train, y_train)
pr1 = test_clf.predict(x_test)

## results
df1 = pd.DataFrame(
        confusion_matrix(y_test, pr1), 
        index = ['ham', 'spam'],
        columns = ['ham', 'spam'],
            )

print(df1)
print(classification_report(y_test, pr1))
print(accuracy_score(y_test, pr1))