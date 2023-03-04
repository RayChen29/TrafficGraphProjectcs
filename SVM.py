
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#Import Data
train_data = pd.read_csv('newtrain.csv')
test_data = pd.read_csv('newtest.csv')

#==================================================
#Arguments: N/A
#Return: feature_train, features to be used
#        features_test, features to be used
#        survival_train, data for train
#        survival_train, data for test
#Description: Sets up the data for the train and test
#             and sets up features. 
#==================================================
def setUp():
    train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1}).astype(int)
    features = train_data[['Age','Sex']]
    survival_rating = train_data['Survived']

    # 20% for testing and 80% for training using sklearn train_test_split lib
    feature_train, feature_test, survival_train, survival_test = train_test_split(features, survival_rating
            , test_size=0.2, random_state=0)

    return feature_train, feature_test, survival_train, survival_test

#==================================================
#Arguments: feature_train, features to be used in train
#           feature_test, features to be used in test
#           survival_train, survival data for train
#           survival_test, survival data for test
#Return:pass
#Description: runs SVM and tests across two different
#             fit types, and multiple degrees
#==================================================
def svm(feature_train, feature_test, survival_train, survival_test):

    kern = ['poly', 'linear']
    
    for i in range(len(kern)):
        if i == 0:
            for k in range(9):
                #poly SVM, ranging from degree 0 to 8
                classifier = SVC(kernel= kern[i], degree= k)
                classifier.fit(feature_train,survival_train)
                score = classifier.score(feature_test, survival_test)
                
                print("++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("Test type: ", kern[i])
                print("Degree: ", k)
                print("Score: ",score)
                print("++++++++++++++++++++++++++++++++++++++++++++++++++")
        else:
                #linear SVM 
                classifier = SVC()
                classifier.fit(feature_train,survival_train)
                score = classifier.score(feature_test, survival_test)
                print("++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("Test type: linear")
                print("Score: ",score)
                print("++++++++++++++++++++++++++++++++++++++++++++++++++")

    pass

 
#========================Main==========================
def main():

    feature_train, feature_test, survival_train, survival_test = setUp()
    svm(feature_train, feature_test, survival_train, survival_test)
   
if __name__ == '__main__':
    main()
