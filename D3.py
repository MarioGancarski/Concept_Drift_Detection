import numpy as np
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression 


def create_model_stream():
    
    return MLPClassifier()

def create_model():
    
    return LogisticRegression(solver='liblinear')


class PredictionManager():
  

    def __init__(self, X, y, new_size, old_size, threshold):
        
        # The index of the current sample
        self.index = 1

        # X : data used, y : labels target of the data
        self.X = X
        self.y = y

        # Model_stream accuracy
        self.accuracy = 0

        # Creating the machine learning models we will use
        self.model_stream = create_model_stream()
        self.model = create_model()

        # List with every possible label for the model targets 
        self.targets = list(set(y))

        self.threshold = threshold

        self.new_size = new_size
        self.old_size = old_size
        self.window_size = old_size + new_size

        self.probs = np.zeros(old_size + new_size)

    #  Run_all_steps function, called by our main function: 

    def run_all_steps(self):
        
        self.model_stream_initial_fit()

        while self.index < len(self.X)-self.window_size :
            
            X_window, y_window = self.X[self.index : self.index + self.window_size], self.y[self.index : self.index + self.window_size]
            
            self.model_stream_predict(X_window, y_window)

            skf = StratifiedKFold(n_splits=2, shuffle=True)
            
            for train_idx, test_idx in skf.split(X_window, y_window):
                
                X_train = list()
                y_train = list()

                for idx in train_idx:
                    X_train.append(X_window[idx])
                    y_train.append(y_window[idx])

                self.model_train(X_train, y_train)

                self.model_predict(X_train, test_idx)
            
            self.check_drift(y_window)
            
            self.model_stream_partial_fit()

            self.index += self.new_size

        self.accuracy /= len(self.y)

        return self.accuracy



    ### Init functions

    def model_stream_initial_fit(self):

        self.model_stream.partial_fit(self.X[0:self.old_size], self.y[0:self.old_size], self.targets)



    ### Run_all_steps loop functions :

    def model_stream_predict(self, X_window, y_window):

        predictions = self.model_stream.predict(X_window[self.old_size:self.window_size])

        for i in range(len(predictions)):

            if predictions[i] == y_window[i]:

                self.accuracy += 1

    def check_drift(self, y_window):

        auc_score = AUC(y_window, self.probs)
        
        print(auc_score)
        #if auc_score > self.threshold:

        #    self.model_stream = create_model_stream()
       
    def model_stream_partial_fit(self):  
        
        self.model_stream.partial_fit(self.X[self.index + self.old_size : self.index + self.window_size], self.y[self.index + self.old_size : self.index + self.window_size], self.targets)



    ### Loop for AUC functions :

    def model_train(self, X_train, y_train): 

        self.model.fit(X_train, y_train)

    def model_predict(self, X_test, test_index):
        
        self.probs[test_index] = self.model.predict_proba(X_test)[:, 1]
