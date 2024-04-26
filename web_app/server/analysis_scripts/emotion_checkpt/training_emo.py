import librosa
import numpy as np
import pandas as pd
import sklearn
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
import copy
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data
import torch.nn.functional as F
import sklearn.model_selection
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
from joblib import dump, load
import xgboost as xgb



def test_xgb_best(X_train, y_train, X_test, y_test):
    # Create the XGBoost classifier
    xgb_model = xgb.XGBClassifier(objective="multi:softprob")
    eval_set = [(X_test, y_test)]


    n_estimators = [100, 200, 300, 400, 500]
    learning_rate = [0.0001, 0.001, 0.01, 0.1]
    param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(xgb_model, param_grid, scoring="accuracy", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train, y_train,  early_stopping_rounds=10, eval_metric="mlogloss", eval_set=eval_set, verbose=False)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))



def RF(X_train, y_train, X_test, y_test):
    # RF
    model_rf = RandomForestClassifier(n_estimators=100,max_depth=10)
    model_rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_rf = model_rf.predict(X_test)

    # Evaluate predictions
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print("Random Forest Accuracy: %.2f%%" % (accuracy_rf * 100.0))


    cm = confusion_matrix(y_test, y_pred_rf, labels=model_rf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=model_rf.classes_)
    disp.plot()
    plt.show()
    plt.savefig("rf_confus_2.png")

def SVM(X_train, y_train, X_test, y_test):
    # Create and train an SVM classifier
    model_svm = SVC(kernel='rbf')
    from joblib import dump, load
    model_svm.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_svm = model_svm.predict(X_test)

    # Evaluate predictions
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print("SVM Accuracy: %.2f%%" % (accuracy_svm * 100.0))
    cm = confusion_matrix(y_test, y_pred_svm, labels=model_svm.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=model_svm.classes_)
    disp.plot()
    plt.show()
    plt.savefig("svm_confus.png")
    dump(model_svm, 'emo_svm.joblib') 

def XGB(X_train, y_train,X_test, y_test, evalset):
    model = xgb.XGBClassifier(
    learning_rate =0.001,
    n_estimators=500,
    max_depth = 5,
    objective="multi:softprob")

    model.fit(X_train, y_train,eval_metric="mlogloss",eval_set=evalset, verbose=False)
    # best_model = grid_result.best_estimator_
    # make predictions for test data
    y_pred = model.predict(X_test)
    prob_list = model.predict_proba(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


    results = model.evals_result()
    
    plt.plot(results['validation_0']['mlogloss'], label='train')
    plt.plot(results['validation_1']['mlogloss'], label='test')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=model.classes_)
    disp.plot()
    plt.show()
    plt.savefig("GBDT_confus_2.png")
    dump(model, 'emo_XGB.joblib') 

if __name__ == "__main__":
    # Load data

    train_data = {"labels": [], "features": []} 
    root_path = "/input/emotion-classification/Training_data/Training_data"
    for folder in os.listdir(root_path):
        label = folder[1]
        folder_path = os.path.join(root_path,folder)
        for files in os.listdir(folder_path):
            feature = np.load(os.path.join(folder_path,files))
            train_data["labels"].append(int(label)-1)
            train_data["features"].append(feature)


    X = np.array(train_data["features"])
    y = np.array(train_data["labels"])
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size=0.1, random_state=42)

    evalset = [(X_train, y_train), (X_test,y_test)]
    
    # test_xgb_best(X_train, y_train, X_test, y_test)
    RF(X_train, y_train, X_test, y_test)
    SVM(X_train, y_train, X_test, y_test)
    XGB(X_train, y_train, X_test, y_test, evalset)