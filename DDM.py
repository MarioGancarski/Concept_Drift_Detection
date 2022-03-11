import math
from models import create_model_stream
from sklearn.metrics import confusion_matrix


class DDM():
  

    def __init__(self, X, y):
        
        # The index of the current sample
        self.index = 1

        # The result of each prediction , 1 = True, 0 = False
        self.prediction = 0

        # The error_rate list used for display
        self.error_rate_list = list()

        # Booleans for drift and warning detection
        self.drift_occured = False
        self.warning_occuring = False

        # X : data used, y : labels target of the data
        self.X = X
        self.y = y

        # The lists of the data in memory during warnings in order to re learn models in case of drift
        self.training_data_X = list()
        self.training_data_y = list()

        # Creating the machine learning model we will use
        self.model = create_model_stream()

        # Creating the concept drift detector instance
        self.cdd = Concept_Drift_Detector()
        self.drift_count = 0

        # List with every possible label for the model targets 
        self.targets = list(set(y))

        # Count of the number of errors by the model prediction
        self.nb_errors = 0

        # This boolean is set to True if the data size for model learning is high enough - I set that to 30
        self.data_size = False
        self.predicts = list()

    # Run_all_steps function, called by our main function: 

    def run_all_steps(self):
        
        self.predicts.append(0)
        while self.index < len(self.X)-1 :

            self.model_train()  
            self.model_predict()    
            self.cdd_step()
            self.check_drift()
            self.check_warning_occuring()
            self.index += 1
        self.predicts.append(0)

        print(confusion_matrix(self.y, self.predicts))

        return self.error_rate_list, 100 * (1 - self.nb_errors / len(self.y))



    ### Run_all_steps loop functions :

    def check_warning_occuring(self):

        if self.warning_occuring or self.data_size:
            self.training_data_X.append(self.X[self.index])
            self.training_data_y.append(self.y[self.index])
        else:
            self.training_data_X = list()
            self.training_data_y = list()


    def check_drift(self):

        if self.drift_occured:
            if len(self.training_data_X)>30 :
                self.model = create_model_stream()
                self.model.partial_fit(self.training_data_X, self.training_data_y, self.targets)
                self.drift_occured = False
                self.data_size = False
                self.drift_count += 1

            else:
                self.data_size = True
        

    def model_train(self): 

        self.model.partial_fit([self.X[self.index]], [self.y[self.index]], self.targets)


    def model_predict(self):
        
        predict = self.model.predict([self.X[self.index+1]])
        self.predicts.append(predict[0])
        if predict[0] == self.y[self.index+1]:
            self.prediction = 0
        else:
            self.prediction = 1
            self.nb_errors += 1



    def cdd_step(self):

        self.new_error_rate, self.drift_occured, self.warning_occuring = self.cdd.cd_detection_step(self.prediction)
        self.error_rate_list.append(self.new_error_rate)
        


class Concept_Drift_Detector():

    '''
    Concept Drift Detection first implementation
    '''

    
    def __init__(self):

        # This is the minimum of probability to have a false prediction which is the error_rate
        self.p_min = 1

        # This is the minimum of the standard deviation
        self.s_min = 1

        # This is the index of the current evaluated sample
        self.s_index = 1

        # This is the boolean giving the information about a warning : if it is true, a warning is currently happening
        self.warning_occuring = False

        # The differents measures and their standard deviation are initialized here
        self.measure = None
        self.s_measure = None

        # The count of each type of prediction are initialized here : TP = "true positive", FP = "false positive", TN = "true negative", FN = "false negative"
        self.nb_errors = 0
        self.size = 0


    ### Cdd functions :

    def check(self):

        # Min check
        if (self.measure + self.s_measure) < (self.p_min +  self.s_min) :
            
            self.p_min = self.measure
            self.s_min = self.s_measure
            self.warning_occuring = False
            self.drift_occured = False
            

        # Check if drift occurs
        elif (self.measure + self.s_measure) > (self.p_min + 3 * self.s_min):

            self.p_min = self.measure
            self.s_min = self.s_measure
            self.nb_errors = 0
            self.size = 0
            self.warning_occuring = False
            self.drift_occured = True


        # Check if warning occurs
        elif (self.measure + self.s_measure) > (self.p_min + 2 * self.s_min):

            self.warning_occuring = True
            self.drift_occured = False


        # When nothing happened and nothing is occuring
        else:

            self.warning_occuring = False
            self.drift_occured = False



    ## This function returns the probability to have error in next prediction = p, the index of the sample (or the index of the beginning of the warning, in case of drift),
    ## and a boolean giving the info if there is drift

    def cd_detection_step(self, value):

        # Increment for each count of predictions
        self.nb_errors += value
        
        self.size += 1

        self.s_index += 1

        self.measure = self.nb_errors/self.size

        self.s_measure = math.sqrt ( self.measure * (1 - self.measure) / ( self.size ) )

        #print(self.measure, self.s_measure)

        self.check()


        return  self.measure , self.drift_occured, self.warning_occuring
