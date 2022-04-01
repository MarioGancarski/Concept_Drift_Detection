import numpy as np
from models import create_model_stream, create_model
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import StratifiedKFold
from skmultiflow.data.data_stream import DataStream



class D3():
  

    def __init__(self, X, y, win_size, rho, threshold, dim):
        
        # The index of the current sample
        self.window_index = 1

        # X : data used, y : labels target of the data
        self.data_stream = DataStream(X, y)
        self.data_stream.prepare_for_use()

        # Old and new sizes for the data windows
        self.new_size = int(win_size * rho)
        self.old_size = int(win_size)
        self.window_size = int(win_size * (1+rho))
        self.rho = float(rho)

        # X_window and y_window
        self.X_window = np.zeros((self.window_size, dim))
        self.y_window = np.zeros(self.window_size)
        
        # Model_stream accuracy
        self.accuracy = 0
        self.predicted_size = 0

        # Creating the machine learning models we will use
        self.model_stream = create_model_stream()
        self.model = create_model()

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
        
        X,y = self.data_stream.next_sample()
        self.addInstance(X,y)
        self.model_stream_partial_fit(X, y)

        while self.data_stream.has_more_samples() :
            
            X,y = self.data_stream.next_sample()

            self.model_stream_predict(X, y)
            self.model_stream_partial_fit(X, y)

            if self.window_index < self.window_size:
                
                """Here we just wait, we don't check if there is a drift because we don't have yet enough data samples"""

            else:

                """ We now have enought data samples, we are going to check if there is a drift"""
                skf = StratifiedKFold(n_splits=2, shuffle=True)
                self.model = create_model()

                for train_idx, test_idx in skf.split(self.X_window, self.y_window):
                    X_train = list()
                    y_train = list()
                    for idx in train_idx:
                        X_train.append(self.X_window[idx])
                        y_train.append(self.y_window[idx])
                    self.model_train(X_train, y_train)
                    self.model_predict(X_train, test_idx)
                 
                if self.check_drift() :
                    #create a self.model_reset function in case reset doesnt exist
                    self.model_stream.reset()
                    self.drift_count += 1
                    self.window_index -= self.old_size

                else :
                    self.window_index -= self.new_size

            self.addInstance(X,y)

        self.accuracy /= self.predicted_size

        print(self.drift_count)
        return self.error_rate_list, self.accuracy



    ### Run_all_steps loop functions :


    def addInstance(self, X, y):
        if(self.window_index < self.window_size):
            self.X_window[self.window_index] = X
            self.y_window[self.window_index] = y
            self.window_index = self.window_index + 1
        else:
            print("Error: Buffer is full!")

    def model_stream_predict(self, X, y):

        prediction = self.model_stream.predict(X)
        if prediction == y:
            self.accuracy += 1
        self.predicted_size += 1 
        self.error_rate_list.append((self.predicted_size - self.accuracy)/self.predicted_size)
       
    def model_stream_partial_fit(self, X, y):  
        
        self.model_stream.partial_fit(X, y, [0,1])

    def check_drift(self):

        auc_score = AUC(self.y_window, self.probs) 
        if auc_score > self.threshold or auc_score < self.threshold - 0.5 :
            return True
        else:
            return False
        

    ### Loop for AUC functions :

    def model_train(self, X_train, y_train): 

        self.model.fit(X_train, y_train)

    def model_predict(self, X_test, test_index):

        self.probs[test_index] = self.model.predict_proba(X_test)[:, 1]
