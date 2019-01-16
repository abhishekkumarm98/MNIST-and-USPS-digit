import pickle
import gzip
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import keras
import sklearn
import tensorflow as tf
from tqdm import tqdm_notebook
%matplotlib inline

# Unzipping the mnist zipped file
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()

# preprocessing
train_x, train_y = training_data
val_x, val_y = validation_data
test_x, test_y = test_data

# For USPS images
imgs = os.listdir("USPSTestSet")
USPS_Test_Imgs = []
digit=[]
for img in imgs:
    img_path = "USPSTestSet/" + img
    if img[-3:]=='png':
        img = Image.open(img_path,'r')
        img = img.resize((28,28))
        digit=img
        img = (255-np.array(img.getdata())) / 255
        USPS_Test_Imgs.append(img)
USPS_Test_Imgs = np.array(USPS_Test_Imgs)
USPS_Test_Imgs.shape   

plt.imshow(digit)

# Creating USPS label
USPS_test_label = []
for i in range(9,-1,-1):
   USPS_test_label +=  [i]*150
USPS_test_label = np.array(USPS_test_label)
USPS_test_label.shape

# Add bias in all samples of MNIST and USPS dataset
train_x=np.concatenate((np.ones((train_x.shape[0],1)),train_x),axis=1)
val_x=np.concatenate((np.ones((val_x.shape[0],1)),val_x),axis=1)
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x),axis=1) 
USPS_Test_Imgs =np.concatenate((np.ones((USPS_Test_Imgs.shape[0],1)),USPS_Test_Imgs),axis=1)

print(train_x.shape)
print(val_x.shape)
print(test_x.shape)
print(USPS_Test_Imgs.shape)

one_hot_train_y = keras.utils.to_categorical(train_y)
one_hot_val_y = keras.utils.to_categorical(val_y)
one_hot_test_y = keras.utils.to_categorical(test_y)
one_hot_USPS_test_label = keras.utils.to_categorical(USPS_test_label)

print(one_hot_train_y.shape)
print(one_hot_val_y.shape)
print(one_hot_test_y.shape)
print(one_hot_USPS_test_label.shape)

def softmaxFunction(inputData, weight):
    intmdTerm = np.exp(np.matmul(inputData, weight) - np.max(np.matmul(inputData, weight)))
    softmax = (intmdTerm.T / np.sum(intmdTerm,axis=1)).T
    return softmax

def cross_entropy(m,data,weight,Target):
    J = (-1/m)*np.sum(Target * np.log(softmaxFunction(data,weight)))
    return J

def gd(m,weight,X, Target, Learningrate,iterations): # gd stands for Gradient Descent
    cost_functionList = [0]* iterations
    for i in tqdm_notebook(range(iterations)):
        weight = weight - (Learningrate/m) * np.dot( X.T, (softmaxFunction(X,weight) - Target))
        cost = cross_entropy(m,X,weight,Target)
        cost_functionList[i] = cost
    return weight, cost_functionList
    
weight = np.zeros((train_x.shape[1],one_hot_test_y.shape[1]))
numOftraining_examples,numOfvalidation_examples,numOftest_examples, numOfUSPS_Test_examples = len(train_x), len(val_x), len(test_x), len(USPS_Test_Imgs)
Learningrate = 0.15
iterations = 300
batch_size = 128
New_weight, costList = gd(numOftraining_examples,weight,train_x,one_hot_train_y,Learningrate,iterations)

plt.plot([i for i in range(iterations)],costList)
plt.xlabel("Number of iterations----->")
plt.ylabel("Cost----->")
plt.title("Cost Vs. Number of iterations")

def mini_batch_sgd(m,weight,X, Target, Learningrate,epochs,batchSize):
    cost_functionList = [0]* epochs
    for epoch in tqdm_notebook(range(epochs)):
        arbit = np.random.permutation(m)
        X = X[arbit]
        Target = Target[arbit]
        cost = 0
        for i in tqdm_notebook(range(0,m,batchSize)):
            X_mini = X[i:i+batchSize]
            Target_mini = Target[i:i+batchSize]
            weight = weight - (Learningrate/m) * np.dot( X_mini.T, (softmaxFunction(X_mini,weight) - Target_mini))
            cost += cross_entropy(m,X,weight,Target)
        cost_functionList[epoch] = cost
    return weight,cost_functionList
    
Weight_now,Cost_list = mini_batch_sgd(numOftraining_examples, weight, train_x, one_hot_train_y, Learningrate, iterations, batch_size)
plt.plot([i for i in range(iterations)],Cost_List,'b.')

def predictions(Input,weight):
    preds = softmaxFunction(Input,weight)
    preds = np.argmax(preds,axis=1)
    return preds
    
def getAccuracy(Input,Target):
    preds = predictions(Input,New_weight) # Use New_weight for Bgd
    accuracy = sum(preds == Target)/(float(len(Target)))
    return accuracy
    
print("Validation Accuracy = ",getAccuracy(val_x,val_y))
print("Test Accuracy = ",getAccuracy(test_x,test_y))

# For USPS test set
print(getAccuracy(USPS_Test_Imgs,USPS_test_label))
predict = predictions(test_x,New_weight)
actualOutput = test_y
print(actualOutput,"\n\n",predict)

df =pd.DataFrame(test_y,columns=['Actual_label'])
df.head()

classified_counts = {}
misclassified_counts ={}

def classificationReport(actual_label, predicted_label):
    for true,pred in list(zip(actual_label, predicted_label)):
        if pred == true:
            if pred in classified_counts:
                classified_counts[pred] = classified_counts[pred]+1
            else:
                classified_counts[pred]=1
        else:
            if pred in misclassified_counts:
                misclassified_counts[pred] = misclassified_counts[pred]+1
            else:
                misclassified_counts[pred]=1
    return classified_counts, misclassified_counts
    
correctCounts, misclassifiedCounts = classificationReport(test_y, predict)
TotalCounts = dict(df['Actual_label'].value_counts())
print("Report:\n")
print(" "*5 + "labels"," "*5 + "precision"," "*5 + "recall" , " "*5 + "f1-score")

for i in range(len(TotalCounts)):
    precision = correctCounts[i]/(correctCounts[i]+misclassifiedCounts[i])
    recall = correctCounts[i]/ TotalCounts[i]
    f1_score = (2 * precision * recall) / (precision + recall)
    print(" "*8,i," "*8,format(precision,".2f")," "*8,format(recall,".2f")," "*8,format(f1_score,".2f"))
    
# Confusion Matrix from sklearn to validate our report's output
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
print("Confusion matrix:\n\n",confusion_matrix(test_y,predict))
print("\naccuracy :\n",accuracy_score(test_y,predict))
print("\nreport :\n",classification_report(test_y,predict))

from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN

clf_1 = Sequential()
# Adding the input layer and the first hidden layer

clf_1.add(Dense(units=64,activation="relu", input_shape=(train_x[:,1:].shape[1],)))

# Adding the output layer

clf_1.add(Dense(units=one_hot_train_y.shape[1], activation= "softmax"))
clf_1.compile(optimizer='sgd',loss= 'categorical_crossentropy',metrics=['accuracy'])
history = clf_1.fit(train_x[:,1:],one_hot_train_y,batch_size=128,epochs=150,verbose=False)

loss_val, accuracy_val = clf_1.evaluate(val_x[:,1:],one_hot_val_y)
loss_test, accuracy_test = clf_1.evaluate(test_x[:,1:],one_hot_test_y)

print("loss for validation MNIST set :",loss_val)
print("Accuracy for validation MNIST set is ",accuracy_val)
print("\nloss for MNIST test set :",loss_test)
print("Accuracy for MNIST test set is ",accuracy_test)

# For USPS set
loss, accuracy = clf_1.evaluate(USPS_Test_Imgs[:,1:],one_hot_USPS_test_label)
print("loss for USPS :",loss)
print("Accuracy for USPS is ",accuracy)

from sklearn.svm import SVC
# Support vector machine classifier

clf_2 = SVC(kernel='linear', C=2, gamma = 0.05)
clf_2.fit(train_x[:,1:], train_y)
print("accuracy :",accuracy_score(val_y,clf_2.predict(val_x[:,1:])))
result = clf_2.predict(test_x[:,1:])

# Confusion Matrix for SVM

print("Confusion matrix:\n\n",confusion_matrix(test_y,result))
print("\naccuracy :\n",accuracy_score(test_y,result))
print("\nreport :\n",classification_report(test_y,result))

# Using rbf kernel
clf_2_ = SVC(kernel='rbf', C=1, gamma = 0.05)
clf_2_.fit(train_x[:,1:], train_y)
print("accuracy :",accuracy_score(val_y,clf_2_.predict(val_x[:,1:])))
result_ = clf_2_.predict(test_x[:,1:])

# Confusion Matrix for SVM

print("Confusion matrix:\n\n",confusion_matrix(test_y,result_))
print("\naccuracy :\n",accuracy_score(test_y,result_))
print("\nreport :\n",classification_report(test_y,result_))

# RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
clf_3 = RandomForestClassifier(n_estimators=10)  
clf_3.fit(train_x[:,1:], train_y)
print("accuracy :",accuracy_score(val_y,clf_3.predict(val_x[:,1:])))
result_rf = clf_3.predict(test_x[:,1:])

# Confusion Matrix for RF

print("Confusion matrix:\n\n",confusion_matrix(test_y,result_rf))
print("\naccuracy :\n",accuracy_score(test_y,result_rf))
print("\nreport :\n",classification_report(test_y,result_rf))

# for USPS

result_rf_ = clf_3.predict(USPS_Test_Imgs[:,1:])
accuracy_score(USPS_test_label,result_rf_)

# Here I have imported Logistic regression from sklearn to 
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(multi_class='ovr')
lr.fit(train_x[:,1:],train_y)
lr.score(test_x[:,1:],test_y)

from sklearn.ensemble import BaggingClassifier
lr_ = lr = LogisticRegression(multi_class='ovr')
clf_2__ = SVC(kernel='rbf', C=1, gamma = 0.05)
clf_3__ = RandomForestClassifier(n_estimators=10)

# Bagging

bagging_1 = BaggingClassifier(base_estimator=lr_, n_estimators=10, max_samples=1)
bagging_2 = BaggingClassifier(base_estimator=clf_2__, n_estimators=10, max_samples=1)
bagging_3 =  BaggingClassifier(base_estimator= clf_3__, n_estimators=10, max_samples=1)

bagging_1.fit(train_x[:,1:],train_y)
bagging_2.fit(train_x[:,1:],train_y)
bagging_3.fit(train_x[:,1:],train_y)
print(bagging_1.score(test_x[:,1:],test_y))
print(bagging_2.score(test_x[:,1:],test_y))
print(bagging_3.score(test_x[:,1:],test_y))


from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier

# Boosting
boosting_1 = AdaBoostClassifier(base_estimator=lr_, algorithm='SAMME')
boosting_2 = AdaBoostClassifier(base_estimator=clf_2__, algorithm='SAMME')
boosting_3 = AdaBoostClassifier(base_estimator=clf_3__, algorithm='SAMME')

boosting_1.fit(train_x[:,1:],train_y)
boosting_2.fit(train_x[:,1:],train_y)
boosting_3.fit(train_x[:,1:],train_y)

print(boosting_1.score(test_x[:,1:],test_y))
print(boosting_2.score(test_x[:,1:],test_y))
print(boosting_3.score(test_x[:,1:],test_y))

from sklearn.ensemble import VotingClassifier

net_clf = VotingClassifier( estimators = [('lr',lr_), ('svm',clf_2__),('rf',clf_3__)], voting = 'hard')
net_clf.fit(train_x[:,1:],train_y)

net_clf_ = VotingClassifier( estimators = [('lr',lr_), ('svm',clf_2__),('rf',clf_3__)], voting = 'soft')
net_clf_.fit(train_x[:,1:],train_y)
print(net_clf.score(test_x[:,1:],test_y))
print(net_clf_.score(test_x[:,1:],test_y))
