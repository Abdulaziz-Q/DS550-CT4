# importing needed packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

#importing datasets
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# check if null values exists
df.info()

# extracting independent and dependent variables
x= df.iloc[:,:-1].values
y= df.iloc[:,12].values # dependent variable is DEATH_EVENT

# spilt the dataset into test and train sets. 80% test 20% train
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=9)

# fitting Random Forest to the dataset
clf = RandomForestClassifier(random_state= 9)
clf = clf.fit(X_train, y_train)

# validate the model

# predict class for X_test
y_pred= clf.predict(X_test)
print(y_pred)

# predict class log-probabilities for X_test
print(clf.predict_log_proba(X_test))

# predict class probabilities for X_test
print(clf.predict_proba(X_test))

# return the mean accuracy on the given test data and labels
print(clf.score(X_test,y_test))

# confusion matrix to determine the correct and incorrect predictions
print(confusion_matrix(y_test, y_pred))
# I got 19+22= 41 incorrect predictions and 144+55= 199 correct predictions