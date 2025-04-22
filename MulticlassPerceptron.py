# author: Sam Hersick
# title: PA-1
# Version: 2.0
# purpose: train a perceptron using one vs. rest to classify wine based on data

#imports
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Perceptron import Perceptron 
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, classification_report

# Model for OVR perceptron
class MulticlassPerceptron:
    #constructor
    def __init__(self, n_iter=600, eta=1.0, random_state=1):
        self.n_iter = n_iter
        self.eta = eta
        self.random_state = random_state

    #train the model
    def train(self, X, y):
        """
        Train a binary perceptron for each class (one-vs-rest).
        :param X: Feature matrix
        :param y: Target labels
        """
        # Gather and Save Classes
        self.classes = []
        for val in y.unique():
            self.classes.append(val)

        #Create and Train All Binary Perceptrons
        self.ptn = {}
        for cl in self.classes:
            ovr_y = np.where(y == cl, 1, 0) # class of y-row is 1 (one) if it is the class this model is testing for, else y-row is 0 (rest)
            self.ptn[cl] = Perceptron(eta=self.eta, n_iter=self.n_iter, random_state=self.random_state) # create binary perceptron for this class
            self.ptn[cl].fit(X, ovr_y) # fit/train perceptron

            #print Errors per Epoch, Weights, Biases
            print(f"\nClass {cl} Perceptron:\n")
            print("Errors: ", self.ptn[cl].errors_)
            print("Weights: ",self.ptn[cl].w_)
            print("Bias: ", self.ptn[cl].b_)

    #predict class of data
    def predict(self, X):
        """
        Predict the class for each sample in X by selecting the perceptron with the highest output.
        :param X: Feature matrix
        :return: List of predicted class labels
        """
        pred = [] # list of predicted classes

        for row in np.asarray(X): # loop through each row/observation
            outputs = {cl: self.ptn[cl].net_input(row) for cl in self.classes} #create a dict: classifier = key, net_input = value
            best_class = max(outputs, key=outputs.get) # calculate best class by finding max value in dict and then taking the key
            pred.append(best_class)

        return pred

# helper methods

#create a box plot of each column to observe outliers
def create_box_plot(df):
    #observe box plot of columns to understand the data
    df.iloc[1:][:].astype(float).plot(kind='box')
    plt.show()

# load the data, drop bad classes, return data split into training and testing
def load_split_data():
    #Get data from wine.data file
    df = pd.read_csv('wine.data', encoding='UTF-8')
    #create_box_plot(df)

    '''observations:
    1. three classes, 1-3
        class 1: row 1 - 59
        class 2: row 60 - 130
        class 3: row 131 - 178
    2. Columns 5 (Magnesium) and 13 (Proline) are on a drastically different scale
    '''
    df.drop(columns=['Magnesium', 'Proline'], inplace=True)
    y = df['Class']
    X = df.drop(columns=['Class'])
    return train_test_split(X, y, random_state=1, test_size=0.5, stratify=y)

# print confusion matrix and classification report
def analysis(y_true, y_pred):
    print("\n\nAnalysis on Test data:\n\n")
    #confusion matrix
    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print("confusion matrix:\n", confmat)

    #classification report
    cl_report = classification_report(y_true=y_true, y_pred=y_pred)
    print("\nclassification report:\n", cl_report)

# main method for running MulticlassPerceptron on our wine data
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split_data()

    #Scale data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #Create and Train Classifier
    mc = MulticlassPerceptron(n_iter=250, eta=0.1)
    mc.train(X_train, y_train)

    #Test Classifier on Test Data
    model_predictions = mc.predict(X_test)
    
    #analyze the results
    analysis(y_test, model_predictions)