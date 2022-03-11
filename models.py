from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression 
from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier

def create_model_stream():
    
    return HoeffdingTreeClassifier()
    #return MLPClassifier()

def create_model():
    
    return LogisticRegression(solver='liblinear')