import numpy as np
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

loan_dataset = pd.read_csv('F:/ML Projects/Loan Status Prediction/dataset.csv')
#print(loan_dataset.head)
#print(loan_dataset.isnull().sum())
loan_dataset=loan_dataset.dropna()

loan_dataset.replace({"Loan_Status":{"N":0,"Y":1}},inplace=True)
loan_dataset.replace({"Dependents":{"3+":4}},inplace=True)
#print(loan_dataset['Dependents'].value_counts())

sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset) #graph for education vs loan_status

loan_dataset.replace({"Married":{"No":0,"Yes":1}},inplace=True)
loan_dataset.replace({"Property_Area":{"Rural":0,"Semiurban":1,"Urban":2}},inplace=True)
loan_dataset.replace({"Education":{"Not Graduate":0,"Graduate":1}},inplace=True)
loan_dataset.replace({"Self_Employed":{"No":0,"Yes":1}},inplace=True)
loan_dataset.replace({"Gender":{"Male":0,"Female":1}},inplace=True)
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']
#print(X)
#print(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
#print(training_data_accuracy)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print(test_data_accuracy)


