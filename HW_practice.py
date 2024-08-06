import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from os import walk
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

#Load data
train_x = []
train_x_std = []
train_y = []
folder_name = ['Yes', 'No']
i = 0
for folder in folder_name:
    path = 'your path'+ str(folder) +'/'
    for root, dirs, files in walk(path):
        for f in files:
            filename = path + f
            print(filename)
            
            
            if folder == 'Yes':    
                train_y.append(1)
                title = 'Original Signal With Chatter #'
                saved_file_name = 'your path'
            
            if folder == 'No':
                train_y.append(0)
                title = 'Original Signal Without Chatter #'
                saved_file_name = 'your path'
                
            # plt.clf()
            plt.figure(figsize=(7,4))
            plt.plot(acc, 'b-', lw=1)
            plt.title(title + str(i+1))
            plt.xlabel('Samples')
            plt.ylabel('Acceleration')
            plt.savefig(saved_file_name + str(i+1) + '.png')                
            # plt.show()
            i = i + 1