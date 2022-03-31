from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from .lookup import numerical_to_categorical, categorical_to_numerical
import pickle


class DataPreProcessor:
    def get_x_y(self, iris, target):
        """Preprocesses the iris dataset

        Args:
            iris (pandas.df): iris dataset which is a pandas dataframe
            target (str): target variable to train the model  on

        Returns:
            numpy.ndarray: x, y numpy arrays
        """
        iris[target] = iris[target].map(categorical_to_numerical)
        x = iris.loc[:, iris.columns != target].to_numpy()
        y = iris[target].to_numpy()
        return x, y

    def split_data(self, x, y, test_size):
        """_summary_

        Args:
            x (numpy.ndarray): dependent variable array
            y (numpy.ndarray): independent variable array
            test_size (float): the size that needs to be split into test

        Returns:
            list: List containing train-test split of inputs.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size)
        return x_train, x_test, y_train, y_test


class SVMClassifier(DataPreProcessor):
    def __init__(self):
        """Initializes the Support Vector Classifier
        """
        self.model = SVC()

    def fit(self, x_train, y_train):
        """Fits x_train and y_train to the initialized model
        """
        self.model.fit(x_train, y_train)

    def predict(self, data):
        """Function to calculate the prediction and lookup of its 
        categorical value

        Args:
            data (list): list containing the dependent variables

        Returns:
            str: the prediction
        """
        prediction = self.model.predict(data).item(0)
        return numerical_to_categorical[prediction]

    def load_model(self, filename):
        """Loads a pickled file

        Args:
            filename (.sav): pickle file name
        """
        self.model = pickle.load(open(filename, 'rb'))
