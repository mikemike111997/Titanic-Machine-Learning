import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class TitanicSolution:

    def get_person_status(self, data_set):
        """
        transforms passengers name into int classes
        :param data_set: DataFrame which is going to be transformed
        :return: transormed DataFrame
        """
        # # insert a new column into DataFrame for saving new feature
        data_set.insert(loc=0, column='Status', value=-999)
        data_set['Status'] = data_set['Name'].map(lambda name: name.split(',')[1].split('.')[0])

        def replace_titles(x):
            """
            function for transorming str into int
            :param x: status of person
            :return:
            """
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

        data_set['Status'] = data_set['Status'].map(lambda x: replace_titles(x))
        return data_set

    def prepare_data(self, file_name='train.csv'):
        """
        read DataFrame and transforms features from str to int
        :param file_name: path to csv file with needed data
        :return: normalized DataFrame
        """
        train_set = pd.read_table(file_name, sep=',')
        # # drop useless data
        train_set = self.get_person_status(train_set)
        train_set.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

        # # encoding data for Embarked
        train_set['Embarked'] = train_set['Embarked'].map({'C': 0.0, 'Q': 1.0, 'S': 2.0})
        # encoding gender
        train_set['Sex'] = train_set['Sex'].map({'female': 1.0, 'male': 0.0})
        train_set.fillna(-99999, inplace=True)
        return train_set

    def get_data_vectors(self):
        """
        :return: test and train vectors for classifiers
        """
        train_set = self.prepare_data()
        y = np.array(train_set['Survived'])
        X = np.array(train_set.drop(['PassengerId', 'Survived'], axis=1), dtype=np.float64)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        return X_train, X_test, y_train, y_test

    def predict_test_samples(self, clf, res_file):
        """
        method for getting predictions from test.csv file and saving result into another csv file
        :param clf: classifier object
        :param res_file: path of result file
        """
        test_df = self.prepare_data('test.csv')
        passenger_id = np.array(test_df['PassengerId'])
        predictions = np.empty(shape=(0,1), dtype=np.int64)
        test_df.drop('PassengerId', axis=1, inplace=True)
        # # loop where we are predicting test samples
        for i, data in test_df.iterrows():
            X = np.array(data, dtype=np.float64)
            X = X.reshape(1, -1)
            prediction = int(clf.predict(X))
            predictions = np.append(predictions, np.array([[prediction]]))
        # # save passengerID and prediction in one np.array
        result = np.vstack((passenger_id, predictions)).T
        result_df = pd.DataFrame(columns=['PassengerId', 'Survived'], data=result)
        # # save DataFrame with passengerID and prediction to a csv file
        result_df.to_csv(path_or_buf=res_file, sep=',', index=False)

    def train_models(self):
        """
        method for training models and doing predictions with that models
        """
        X_train, X_test, y_train, y_test = self.get_data_vectors()

        self.model_SVC = SVC()
        self.model_SVC.fit(X_train, y_train)
        self.accuracy_SVC = self.model_SVC.score(X_test, y_test)
        print('accuracy_SVC = {}'.format(self.accuracy_SVC))
        self.predict_test_samples(self.model_SVC, 'model_SVC.csv')


        self.model_KN = KNeighborsClassifier()
        self.model_KN.fit(X_train, y_train)
        self.accuracy_KN = self.model_KN.score(X_test, y_test)
        print('accuracy_KN = {}'.format(self.accuracy_KN))
        self.predict_test_samples(self.model_KN, 'model_KN.csv')

        self.model_decision_tree = DecisionTreeClassifier()
        self.model_decision_tree.fit(X_test, y_test)
        accuracy_DT = self.model_decision_tree .score(X_test, y_test)
        print('accuracy_DT = {}'.format(accuracy_DT))
        self.predict_test_samples(self.model_decision_tree, 'model_decision_tree.csv')

if __name__ == '__main__':
    solution = TitanicSolution()
    solution.train_models()