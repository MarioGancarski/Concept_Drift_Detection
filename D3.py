import numpy as np
from models import create_model_stream, create_model
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import StratifiedKFold



class D3():
  

    def __init__(self, X, y, win_size, rho, threshold):
        
        # The index of the current sample
        self.index = 0

        # X : data used, y : labels target of the data
        self.X = X
        self.y = y

        # Old and new sizes for the data windows
        self.new_size = int(win_size * rho)
        self.old_size = int(win_size)
        self.window_size = int(win_size * (1+rho))

        # Model_stream accuracy
        self.accuracy = 0
        self.predicted_size = 0

        # Creating the machine learning models we will use
        self.model_stream = create_model_stream()
        self.model = create_model()

        # List with every possible label for the model targets 
        self.targets = list(set(y))

        # Threshold for the auc measure detection
        self.threshold = threshold

        # Array with prediction probabilities
        self.probs = np.zeros(self.window_size)

        # Drift count
        self.drift_count = 0

        # error rate list
        self.error_rate_list = list()

    #  Run_all_steps function, called by our main function: 

    def run_all_steps(self):
        
        while self.index < len(self.X)-2*self.window_size :
            
            
            X_window, y_window = self.X[self.index : self.index + self.window_size], self.y[self.index : self.index + self.window_size]
            skf = StratifiedKFold(n_splits=2, shuffle=True)
            self.model_stream_partial_fit()
            self.model_stream_predict(X_window, y_window)
            self.model = create_model()

            for train_idx, test_idx in skf.split(X_window, y_window):
                X_train = list()
                y_train = list()
                for idx in train_idx:
                    X_train.append(X_window[idx])
                    y_train.append(y_window[idx])
                self.model_train(X_train, y_train)
                self.model_predict(X_train, test_idx)
            
            if self.check_drift(y_window):
                self.index += self.old_size
                self.drift_count += 1

            else :
                self.index += self.new_size

        self.accuracy /= self.predicted_size

        return self.error_rate_list, self.accuracy



    ### Init functions

    def model_stream_initial_fit(self):

        self.model_stream.partial_fit(self.X[0:self.old_size], self.y[0:self.old_size], self.targets)



    ### Run_all_steps loop functions :

    def model_stream_predict(self, X_window, y_window):

        predictions = self.model_stream.predict(X_window[self.old_size:self.window_size])
        for i in range(len(predictions)):
            if predictions[i] == y_window[i]:
                self.accuracy += 1
            self.predicted_size += 1 
            self.error_rate_list.append((self.predicted_size - self.accuracy)/self.predicted_size)


    def check_drift(self, y_window):

        auc_score = AUC(y_window, self.probs) 
        if auc_score > self.threshold or auc_score < self.threshold - 0.5 :
            self.model_stream = create_model_stream()
            return True
        else:
            return False
        
       
    def model_stream_partial_fit(self):  
        
        self.model_stream.partial_fit(self.X[self.index + self.old_size : self.index + self.window_size], self.y[self.index + self.old_size : self.index + self.window_size], self.targets)



    ### Loop for AUC functions :

    def model_train(self, X_train, y_train): 

        self.model.fit(X_train, y_train)

    def model_predict(self, X_test, test_index):

        self.probs[test_index] = self.model.predict_proba(X_test)[:, 1]
