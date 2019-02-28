# Useful Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC


'''
    Author : Mathieu VANDECASTEELE - mathieuvdc.com

    Perform a SVC (SVM) for Classification. Most of the cases, you just need to put X and y for input parameters.
    
    Return : SVC() classifier fit with the best parameters.
    
    Parameters:
    X : data table with a shape (n, m).
    y : labels of the data X with a shape (n,).
    test_size : the ratio for splitting the dataset.
    standardize : Choose if you want to standardize the data.
    parameters_grid : dictionnary of the GridSearchCV parameters.
    cv_number : number of cross-validation
    njobs_gs : number of jobs for GridSearchCV
    display_gs_details : Set to True if you want to display all the scores for every combinations of parameters in the GridSearchCV.
    random_state : Set the random_state of the splitting.
    
    
'''

def my_SVC(X, y, test_size = 0.20, standardize = True, parameters_grid = {'kernel':('rbf','linear'), 'C':[0.001,0.01,0.1,1,2,5,10,15,20]}, cv_number = 5, njobs_gs = -1, display_gs_details = False, random_state = 42):
    
    print("\n----------BEGIN SVC() FOR CLASSIFICATION----------\n")
    print("Shape of X is "+str(X.shape))
    print("Shape of y is "+str(y.shape))

    # Split the Data
    test_size = test_size
    random_state = random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("\nSpliting...")
    print("Test ratio : "+str(test_size))
    print("Shape of Train dataset is "+str(X_train.shape))
    print("Shape of Test dataset is "+str(X_test.shape))
    
    # Standardization
    # Set to True or False to standardize the data.
    standardize = True
    
    if standardize :
        print("\nStandardization ...")
        scaler = StandardScaler().fit(X_train)                                
        X_train = scaler.transform(X_train)                           
        X_test = scaler.transform(X_test)   
    
    # First Simple Training
    print("\n--------------------------------------------------")
    print("\nPerform First Simple Training...\n")
    svc = SVC(gamma="scale")
    print(svc)

    time1 = time.time()
    svc.fit(X_train, y_train)
    elapsed = time.time() - time1
    print("\nDone ! Time for training : "+str(elapsed)+" seconds.\n") 

    # Simple Training Report
    score = svc.score(X_test, y_test)
    print("Precision/Score on Test Dataset : "+str(score)+"\n")

    y_pred = svc.predict(X_test)
    print("Classification Report on Test Dataset (y_pred/y_test) :\n")
    print(classification_report(y_pred, y_test))
    print("Confusion Matrix on Test Dataset (y_pred/y_test) :\n")
    print(confusion_matrix(y_pred, y_test))    
    
    # Tuning parameters with GridSearch and Cross-Validation 
    print("\n--------------------------------------------------")
    print("\nPerform GridSearchCV and tuning parameters... \n")
    
    # Personalization of the parameters : 
    parameters = parameters_grid
    cv_number = cv_number
    # n_jobs for parallel computing. Set to -1 for using all the processors.
    njobs = njobs_gs
    
    svc = SVC(gamma="scale")
    gs = GridSearchCV(svc, parameters, cv=cv_number)
    print(gs)
    
    time2 = time.time()
    gs.fit(X_train,y_train)
    elapsed2 = time.time() - time2
    print("\nDone ! Time elapsed for tuning parameters : "+str(elapsed2)+" seconds.\n") 
    
    
    # GridSearchCV Report

    print("Best validation score : "+str(gs.best_score_))
    
    print("\nBest parameters : ")
    best_parameters = gs.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    score = gs.score(X_test, y_test)
    print("\nPrecision/Score on Test Dataset : "+str(score)+"\n")
    
    y_pred = gs.predict(X_test)
    print("Classification Report on Test Dataset (y_pred/y_test) :\n")
    print(classification_report(y_pred, y_test))
    print("Confusion Matrix on Test Dataset (y_pred/y_test) :\n")
    print(confusion_matrix(y_pred, y_test))
    print("\nAverage training time for one model :") 
    print(str(np.mean(gs.cv_results_['mean_fit_time']))+" seconds.")


    # All results ("Set to True to see all results forom GridSearchCV")
    display_gridsearch_details = display_gs_details

    if display_gridsearch_details :
        print('All results :')
        means = gs.cv_results_['mean_test_score']
        stds = gs.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gs.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print('')
        
    # Last Final Training
    print("\n--------------------------------------------------")
    print("\nPerform Last Final Training...\n")
    
    # For parameters you can directly unpack the best params dictionary of the GridSearchCV classifier or write it manually.
    svc = SVC(gamma='scale', **gs.best_params_)
    print(svc)
    
    time3 = time.time()
    svc.fit(X_train, y_train)
    elapsed3 = time.time() - time3
    print("\nDone ! Time for training : "+str(elapsed3)+" seconds.\n") 

    # Last Training Report
    score = svc.score(X_test, y_test)
    print("Precision/Score on Test Dataset : "+str(score)+"\n")
    
    y_pred = svc.predict(X_test)
    print("Classification Report on Test Dataset (y_pred/y_test) :\n")
    print(classification_report(y_pred, y_test))
    print("Confusion Matrix on Test Dataset (y_pred/y_test) :\n")
    print(confusion_matrix(y_pred, y_test))
    print("\n----------DONE----------\n")
    
    return svc