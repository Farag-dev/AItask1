
#import Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#get DataSet
fruits = pd.read_csv("H:\\Education\College\\04 Final year\\AI 2\\ML\\Lab 02\\Python File\\fruit_data_with_colours.csv")

fruits.head()

Fruit_name =  dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique() ))

print(type(Fruit_name))
print(Fruit_name)

# (Features)
X = fruits[['mass','width','height']]
X.head(10)

# (Response)
y= fruits['fruit_label']
y.head(10)


#Split data 
X_train,X_test,y_train,y_test =  train_test_split(X,y,random_state=42)


#Classifier by KNeighbors

from sklearn.neighbors import KNeighborsClassifier

# added
# accuracy method of KNeighbors
# take (test_label, predicted_values)
from sklearn.metrics import accuracy_score


ACs = []
for i in range(1, 16):
    
    k = KNeighborsClassifier(n_neighbors=i)

    #train_data

    k.fit(X_train,y_train)

    #test_data

    k.score(X_test,y_test)

    #Get predicted response 
    y_predict = k.predict(X_test)

    # added
    Ac = accuracy_score(y_test, y_predict)
    ACs.append(Ac)
    print(f"Accuracy of {i} = {Ac}")

    
print() 
print(f"Max accuracy is in k = {ACs.index(max(ACs)) + 1} \n")
