import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def get_person_status(train_set):
    train_set.insert(loc=0, column='Status', value=-999)
    train_set['Status'] = train_set['Name'].map(lambda name: name.split(',')[1].split('.')[0])

    def replace_titles(x):
        if x in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 0#'Mr'
        elif x in ['Countess', 'Mme']:
            return 1#'Mrs'
        elif x in ['Mlle', 'Ms']:
            return 2#'Miss'
        elif x == 'Dr':
            return 3#'Dr'
        else:
            return 4#title

    train_set['Status'] = train_set['Status'].map(lambda x: replace_titles(x))
    return train_set
    # print(set(list(train_set['Status'])))


def prepare_data():
    train_set = pd.read_table('train.csv', sep=',')
    # drop useless data
    train_set = get_person_status(train_set)
    train_set.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

    # encoding data for Embarked
    train_set['Embarked'] = train_set['Embarked'].map({'C': 0.0, 'Q': 1.0, 'S': 2.0})
    # encoding gender
    train_set['Sex'] = train_set['Sex'].map({'female': 1.0, 'male': 0.0})

    train_set.fillna(-99999, inplace=True)
    return train_set


def get_data_vectors():
    train_set = prepare_data()
    print(train_set.head(2))
    X = np.array(train_set['Survived'])
    y = np.array(train_set.drop(['Survived'], axis=1), dtype=np.float64)
    return train_test_split(X, y, test_size=0.2)


def train_models():
    X_train, X_test, y_train, y_test = get_data_vectors()

    model_SVC = SVC()
    model_SVC.fit(X_train, [-1 if int(i) == 0 else 1 for i in y_train])
    accuracy_SVC = model_SVC.score(X_test, [-1 if int(i) == 0 else 1 for i in y_test])
    print(accuracy_SVC)

    model_KN = KNeighborsClassifier()
    model_KN.fit(X_train, y_train)
    accuracy_KN = model_KN.score(X_test, y_test)
    print(accuracy_KN)

train_models()