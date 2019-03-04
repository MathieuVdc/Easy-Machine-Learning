# Useful Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score


'''
    Author : Mathieu VANDECASTEELE - mathieuvdc.com

    Perform a GradientBoosting for Regression. Most of the cases, you just need to put X and y for input parameters.
    
    Return : GradientBoostingRegressor() fit with the best parameters.
    
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

def my_GBRegressor(X, y, test_size = 0.20, standardize = True, parameters_grid = {'n_estimators':[50,100,150,200,250,300], 'learning_rate':[0.1,0.1125,0.125,0.15],'max_depth':[3,4,5,6,7]}, cv_number = 5, njobs_gs = -1, display_gs_details = False, random_state = 42):
    
    print("\n----------BEGIN GradientBoostingRegressor()----------\n")
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
    gb = GradientBoostingRegressor()
    print(gb)

    time1 = time.time()
    gb.fit(X_train, y_train)
    elapsed = time.time() - time1
    print("\nDone ! Time for training : "+str(elapsed)+" seconds.\n") 

    # Simple Training Report
    score = gb.score(X_test, y_test)
    print("Precision/Score on Test Dataset : "+str(score)+"\n")

    y_pred = gb.predict(X_test)
    print("Regression Metrics :\n")
    print("Explained variance : "+str(explained_variance_score(y_test, y_pred)))
    print("Mean Absolute Error : "+str(mean_absolute_error(y_test, y_pred)))
    print("Mean Squared Error : "+str(mean_squared_error(y_test, y_pred)))
    print("Mean Squared Log Error : "+str(mean_squared_log_error(y_test, y_pred)))
    print("Median Absolute Error : "+str(median_absolute_error(y_test, y_pred)))
    print("R2 Score : "+str(r2_score(y_test, y_pred)))
    
    plt.figure(figsize=(20,8))
    plt.plot(y_test, c="r",label='true')
    plt.plot(y_pred,c='g',label='prediction')
    plt.ylabel('y-value')
    plt.title('First Training : True Test Dataset vs Prediction Test Dataset')
    plt.legend()
  
    
    # Tuning parameters with GridSearch and Cross-Validation 
    print("\n--------------------------------------------------")
    print("\nPerform GridSearchCV and tuning parameters... \n")
    
    # Personalization of the parameters : 
    parameters = parameters_grid
    cv_number = cv_number
    # n_jobs for parallel computing. Set to -1 for using all the processors.
    njobs = njobs_gs
    
    gb = GradientBoostingRegressor()
    gs = GridSearchCV(gb, parameters, cv=cv_number)
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
    print("Regression Metrics :\n")
    print("Explained variance : "+str(explained_variance_score(y_test, y_pred)))
    print("Mean Absolute Error : "+str(mean_absolute_error(y_test, y_pred)))
    print("Mean Squared Error : "+str(mean_squared_error(y_test, y_pred)))
    print("Mean Squared Log Error : "+str(mean_squared_log_error(y_test, y_pred)))
    print("Median Absolute Error : "+str(median_absolute_error(y_test, y_pred)))
    print("R2 Score : "+str(r2_score(y_test, y_pred)))
    
    print("\nAverage training time for one model :") 
    print(str(np.mean(gs.cv_results_['mean_fit_time']))+" seconds.")


    # All results ("Set to True to see all results forom GridSearchCV")
    display_gridsearch_details = False
    
    if display_gridsearch_details :
        print('All results :')
        means = gs.cv_results_['mean_test_score']
        stds = gs.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gs.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        
    # Last Final Training
    print("\n--------------------------------------------------")
    print("\nPerform Last Final Training...\n")
    
    # For parameters you can directly unpack the best params dictionary of the GridSearchCV classifier or write it manually.
    gb = GradientBoostingRegressor(**gs.best_params_)
    print(gb)
    
    time3 = time.time()
    gb.fit(X_train, y_train)
    elapsed3 = time.time() - time3
    print("\nDone ! Time for training : "+str(elapsed3)+" seconds.\n") 

    # Last Training Report
    score = gb.score(X_test, y_test)
    print("Validation score calculated previously for this model : "+str(gs.best_score_))
    print("Precision/Score on Test Dataset : "+str(score)+"\n")
    
    y_pred = gb.predict(X_test)
    print("Regression Metrics :\n")
    print("Explained variance : "+str(explained_variance_score(y_test, y_pred)))
    print("Mean Absolute Error : "+str(mean_absolute_error(y_test, y_pred)))
    print("Mean Squared Error : "+str(mean_squared_error(y_test, y_pred)))
    print("Mean Squared Log Error : "+str(mean_squared_log_error(y_test, y_pred)))
    print("Median Absolute Error : "+str(median_absolute_error(y_test, y_pred)))
    print("R2 Score : "+str(r2_score(y_test, y_pred)))
    
    plt.figure(figsize=(20,8))
    plt.plot(y_test, c="r",label='true')
    plt.plot(y_pred,c='g',label='prediction')
    plt.ylabel('y-value')
    plt.title('Final Training : True Test Dataset vs Prediction Test Dataset')
    plt.legend()
    
    return gb