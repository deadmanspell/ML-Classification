# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

#Taking_Data_from_Directory
print("Predicting Iris Flower Species Using Random Forest Algorithm")
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ["sepal_length","sepal_width","petal_length","petal_width","species"]
data = pd.read_csv(url, names = names)
data.columns = names
print("Iris Species: ", data["species"].unique())
print(data["species"].value_counts())

#Encoding_the_Output
le =  LabelEncoder()
label_mapping = {
    "Iris-setosa":0,
    "Iris-versicolor":1,
    "Iris-virginica":2
    }
data['species'] = data['species'].map(label_mapping)

#Splitting_Data
X_train, X_test, y_train, y_test = train_test_split(
    data[data.columns[:-1]], data["species"], test_size=0.10
)

#Random_Forest_Model_From_Splitted_Data
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

#Making_Prediction
print("Prediction for row 2: ", clf.predict([[4.9, 3, 1.4, 0.2]]))
print("Prediction probability for row 2: ", clf.predict_proba([[4.9, 3, 1.4, 0.2]]))
print(clf.predict(X_test))
print(y_test)

#Showing_Pred_Accuracy
print("Prediction Accuracy for Train: ", clf.score(X_train, y_train))

#Showing_Pred_In_Chart
predicted = clf.predict(X_test)
result = pd.DataFrame({"Predicted": predicted, "Target": np.array(y_test)})
print("Prediction Accuracy for Test: ", clf.score(X_test, y_test))
filtro = result.Predicted == result.Target
print(filtro.value_counts(normalize=True))

#Plotting_With_Confusion_Matrix
df = pd.DataFrame(result)
confusion_matrix = pd.crosstab(df['Predicted'], df['Target'], rownames=['Target'], colnames=['Predicted'], margins = True)
confusion_matrix

sns.heatmap(confusion_matrix, annot=True)
plt.show()


