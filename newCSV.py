import numpy as np
import pandas as pd
from svmCleaner import driver

train_data = np.loadtxt("./train.csv", 
                        delimiter=",", skiprows=1, dtype = str)
test_data = np.loadtxt("./test.csv", 
                        delimiter=",", skiprows=1, dtype = str)
#initially write fields
newtrain = open('newtrain.csv','w')
newtest = open('newtest.csv','w')

newtrain.close()
newtest.close()

#go back and write data in.
newtrain = open('newtrain.csv','a')
newtest = open('newtest.csv','a')

trainfields = (['PassengerId,','Survived,','Pclass,','Sex,','Age,','SibSp,','Parch,','Ticket,','Fare,','Cabin,','Embarked,\n'])
testfields = (['PassengerId,','Pclass,','Sex,','Age,','SibSp,','Parch,','Ticket,','Fare,','Cabin,','Embarked,\n'])

newtrain.writelines(trainfields)
newtest.writelines(testfields)
print("Pre Driver")
train_data, test_data = driver(train_data,test_data) #I'm assuming this just lets me officially work with the data.
print("Post Driver")
#write in training data
for i in range(len(train_data)):
    line = [train_data[i][0]]
    for j in range(len(train_data[i])):
        line.append(train_data[i][j]+",")
    newtrain.writelines(line+['\n'])
    
#write in testing data    
for i in range(len(test_data)):
    line = [test_data[i][0]]
    for j in range(len(test_data[i])):
        line.append(test_data[i][j]+",")
    newtest.writelines(line+['\n'])
newtrain.close()
newtest.close()
