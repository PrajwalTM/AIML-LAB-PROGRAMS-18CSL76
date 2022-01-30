PROGRAM.NO.7 (Naive bayes Classifier)
"""" Write a program to implement the naïve Bayesian classifier for a sample training data set stored as a .CSV file. 
Compute the accuracy of the classifier, considering few test data sets. """


import pandas as pd
msg=pd.read_csv('naivetext1.csv',names=['message','label'])
print('The dimensions of the dataset',msg.shape)
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
y=msg.labelnum
#print(X)
#print(y)
#splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,y)
print('dimensions of train and test sets')
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

#output of count vectoriser is a sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
#print(count_vect.vocabulary_)
xtest_dtm=count_vect.transform(xtest)

# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm,ytrain)
#print(clf)
predicted = clf.predict(xtest_dtm)
#printing accuracy metrics
from sklearn import metrics
print('Accuracy metrics')
print('Accuracy of the classifer is',metrics.accuracy_score(ytest,predicted))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('Recall and Precison ')
print(metrics.recall_score(ytest,predicted))
print(metrics.precision_score(ytest,predicted))
      
########################################################################################################################
# OUTPUT:
# Ignore single quotes at beginning and end
########################################################################################################################
      
The dimensions of the dataset (18, 2)
dimensions of train and test sets
(13,)
(5,)
(13,)
(5,)
Accuracy metrics
Accuracy of the classifer is 0.6
Confusion matrix
[[1 1]
[1 2]]
Recall and Precison
0.6666666666666666
0.6666666666666666
