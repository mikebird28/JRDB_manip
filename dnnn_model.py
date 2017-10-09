
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import sqlite3
import feature
import dataset2
import util
import evaluate


def main():
    config = util.get_config("config/config.json")
    #generate dataset
    db_path = "db/output_v5.db"
    pca = PCA(n_components = 10)

    print(">> loading dataset")
    x,y = dataset2.load_dataset(db_path,config.features,"win")
    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y)

    print(">> filling none value of train dataset")
    #train_x = dataset2.fillna_mean(train_x,"race")
    train_x = dataset2.fillna_mean(train_x,"horse")
    mean = train_x.mean(numeric_only = True)
    std = train_x.std(numeric_only = True)
    train_x = dataset2.normalize(train_x,mean = mean,std = std)
    #train_x = dataset2.normalize(train_x,typ = "race")


    print(">> generating train pca dataset")
    pca_x,pca_y = dataset_for_pca(train_x,train_y)
    pca_idx = pca_x["info_race_id"]
    pca_x,pca_y = dataset2.for_use(pca_x,pca_y)

    print(">> fitting with pca")
    pca.fit(pca_x)
    print(sum(pca.explained_variance_ratio_))
    print(pca.explained_variance_ratio_)
    pca_df = pd.DataFrame(pca.transform(pca_x))
    pca_df = pd.concat([pd.DataFrame(pca_df),pca_idx],axis = 1)
    train_x,train_y = dataset2.add_race_info(train_x,train_y,pca_df)


    print(">> under sampling train dataset")
    train_x,train_y = dataset2.under_sampling(train_x,train_y)
    train_x,train_y = dataset2.for_use(train_x,train_y)

    print(">> filling none value of test dataset")
    #test_x = dataset2.fillna_mean(test_x,"race")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.normalize(test_x,mean = mean,std = std)

    print(">> generating test pca dataset")
    pca_x,pca_y = dataset_for_pca(test_x,test_y,mean = mean,std = std)
    pca_idx = pca_x["info_race_id"]
    pca_x,pca_y = dataset2.for_use(pca_x,pca_y)
    pca_df = pca.transform(pca_x)
    pca_df = pd.concat([pd.DataFrame(pca_df),pca_idx],axis = 1)
    test_x,test_y = dataset2.add_race_info(test_x,test_y,pca_df)

    print(">> under sampling test dataset")
    test_rx,test_ry = dataset2.to_races(test_x,test_y)
    test_x,test_y = dataset2.under_sampling(test_x,test_y)
    test_x,test_y = dataset2.for_use(test_x,test_y)

    dnn(config.features,train_x,train_y,test_x,test_y,test_rx,test_ry)
    #dnn_wigh_gridsearch(train_x.columns,train_x,train_y,test_x,test_y,test_rx,test_ry)

def dnn(features,train_x,train_y,test_x,test_y,test_rx,test_ry):
    mlpc = MLPClassifier(
        hidden_layer_sizes = (50,50),
        solver = "adam",
        batch_size = 300,
        learning_rate_init = 0.1,
        epsilon = 1e-4,
        verbose = 10,
        tol = 0.1e-5
    )
    mlpc.fit(train_x,train_y)

    pred = mlpc.predict(test_x)
    accuracy = accuracy_score(test_y,pred)
    top_1_k  = evaluate.top_n_k(mlpc,test_rx,test_ry)
    report = classification_report(test_y,pred)
    print("")
    print("Accuracy: {0}".format(accuracy))
    print("top_1_k : {0}".format(top_1_k))
    print(report)

def dnn_wigh_gridsearch(features,train_x,train_y,test_x,test_y,test_rx,test_ry):
    paramaters = [
        {'n_estimators':[100],
        'learning_rate':[0.05],
        'max_depth' : [5],
        'subsample':[0.5,0.6,0.7],
        'min_child_weight':[0.8,1.0,1.2]}
#        {'colsample_bytree':[0.8,1.0]},
    ]

    mlpc = MLPClassifier()
    cv = GridSearchCV(mlpc,paramaters,cv = 2,scoring='accuracy',verbose = 2)
    cv.fit(train_x,train_y)
    pred = cv.predict(test_x)

    accuracy = accuracy_score(test_y,pred)
    top_1_k  = evaluate.top_n_k(cv,test_rx,test_ry)
    report = classification_report(test_y,pred)
    print("")
    print("Accuracy: {0}".format(accuracy))
    print("top_1_k : {0}".format(top_1_k))
    print(report)


    print("Paramaters")
    #best_parameters, score, _ = max(cv.grid_scores_, key=lambda x: x[1])
    best_parameters = cv.best_params_
    print(best_parameters)
    for pname in sorted(best_parameters.keys()):
        print("{0} : {1}".format(pname,best_parameters[pname]))

def dataset_for_pca(x,y,mean = None,std = None):
    #x = dataset2.normalize(x,mean = mean,std = std)
    x,y = dataset2.pad_race(x,y)
    x = dataset2.flatten_race(x)
    return (x,y)

if __name__=="__main__":
    main()