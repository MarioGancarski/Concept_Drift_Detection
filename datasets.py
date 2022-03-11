import csv
import numpy as np

def import_dataset(dataset_name):
    X = list()
    y = list()
    first = True
    with open(dataset_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar ='|')
        for row in spamreader:
          
            if first :
                first = False
            else:
                elem = list()
                for i in range (len(row)-1):
                    elem.append(float(row[i].replace(',', '.')))
                X.append(elem)
                y.append(float(row[len(row)-1].replace(',', '.')))
    return X, y


