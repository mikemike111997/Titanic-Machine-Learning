import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
train_set = pd.read_table('train.csv', sep=',')
# drop useless data
train_set.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)


# encoding data for Embarked
train_set['Embarked'] = train_set['Embarked'].map({'C': 0.0, 'Q': 1.0, 'S': 2.0})
# encoding gender
train_set['Sex'] = train_set['Sex'].map({'female': 1.0, 'male': 0.0})

train_set.fillna(-99999, inplace=True)

y = np.array(train_set['Survived'])
X = np.array(train_set.drop(['Survived'], axis=1), dtype=np.float64)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model_SVC = SVC()
model_SVC.fit(X_train, [-1 if int(i) == 0 else 1 for i in y_train])
accuracy_SVC = model_SVC.score(X_test, [-1 if int(i) == 0 else 1 for i in y_test])
print(accuracy_SVC)

model_KN = KNeighborsClassifier()
model_KN.fit(X_train, y_train)
accuracy_KN = model_KN.score(X_test, y_test)
print(accuracy_KN)

