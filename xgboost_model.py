
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import xgboost as xgb
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
    db_path = "db/output_v3.db"
    pca = PCA(n_components = 1)

    print(">> loading dataset")
    x,y = dataset2.load_dataset(db_path,config.features,"win")
    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y)

    print(train_y.count())
    print(train_x.count())
 
    print(">> filling none value of train dataset")
    #train_x = dataset2.fillna_mean(train_x,"race")
    train_x = dataset2.fillna_mean(train_x,"horse")

    print(">> generating train pca dataset")
    pca_x,pca_y = dataset_for_pca(train_x,train_y)
    pca_idx = pca_x["info_race_id"]
    pca_x,pca_y = dataset2.for_use(pca_x,pca_y)

    print(">> fitting with pca")
    pca.fit(pca_x)
    pca_df = pd.DataFrame(pca.transform(pca_x))
    pca_df = pd.concat([pd.DataFrame(pca_df),pca_idx],axis = 1)
    train_x = train_x.merge(pca_df,on = "info_race_id",how="left")
    print(train_y.count())
    print(train_x.count())

    print(">> under sampling train dataset")
    train_x,train_y = dataset2.under_sampling(train_x,train_y)
    print(train_x)
    print(train_y)
    train_y = dataset2.fillna_zero(train_y)
    train_x,train_y = dataset2.for_use(train_x,train_y)

    print(">> filling none value of test dataset")
    #test_x = dataset2.fillna_mean(test_x,"race")
    test_x = dataset2.fillna_mean(test_x,"horse")
    print(">> generating test pca dataset")
    pca_x,pca_y = dataset_for_pca(test_x,test_y)
    pca_idx = pca_x["info_race_id"]
    pca_x,pca_y = dataset2.for_use(pca_x,pca_y)
    pca_df = pca.transform(pca_x)
    pca_df = pd.concat([pd.DataFrame(pca_df),pca_idx],axis = 1)
    test_x = test_x.merge(pca_df,on = "info_race_id",how = "left")
    print(">> under sampling test dataset")
    test_rx,test_ry = dataset2.to_races(test_x,test_y)
    #test_rx,test_ry = dataset2.for_use(test_rx,test_ry)
    test_x,test_y = dataset2.under_sampling(test_x,test_y)
    test_y = dataset2.fillna_zero(test_y)
    test_x,test_y = dataset2.for_use(test_x,test_y)

    xgbc(train_x,train_y,train_x,train_y,test_rx,test_ry)
    #xgbc(train_x,train_y,test_x,test_y,test_rx,test_ry)

    """
    pca.fit(ptrain_x)
    pca_train_x = pca.transform(ptrain_x)
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))

    train_x = dataset.add_race_info(train_x,pca_train_x)
    train_x,train_y = dataset.races_to_horses(train_x,train_y)
    train_x,train_y = preprocess(train_x,train_y)
    print(">> Training data preprocessing finished")

    #preprocess for test data

    #grid search

    paramaters = [
        {'max_depth' : [10]},
        {'min_child_weight':[0.6,0.8,1.0]},
        {'subsample':[0.6,0.8,1.0]},
        {'colsample_bytree':[0.6,0.8,1.0]},
    ]
    cv = GridSearchCV(xgbc,paramaters,cv = 3,scoring='accuracy',verbose = 2)

    #best_forest = cv.best_estimator_
    #cv.fit(train_x,train_y)
    #pred = cv.predict(test_hx)

    #best = cv.best_estimator_
    #print(best)

    print("Top 1   : {0}".format(evaluate.top_n_k(xgbc,config.features,test_rx,test_ry)))
    print("Accuracy: {0}".format(accuracy_score(test_hy,pred)))
    print(classification_report(test_hy,pred))
    #importances = best.feature_importances_
    importances = xgbc.feature_importances_
    #evaluate.plot_importance(config.features,importances)
    evaluate.show_importance(config.features,importances)
    """

def xgbc(train_x,train_y,test_x,test_y,test_rx,test_ry):
    xgbc = xgb.XGBClassifier()
    xgbc.fit(train_x,train_y)

    pred = xgbc.predict(test_x)
    accuracy = accuracy_score(test_y,pred)
    top_1_k  = evaluate.top_n_k(xgbc,test_rx,test_ry)
    report = classification_report(test_y,pred)
    print("")
    print("Accuracy: {0}".format(accuracy))
    print("top_1_k : {0}".format(top_1_k))
    print(report)


def dataset_for_pca(x,y):
    x,y = dataset2.pad_race(x,y)
    x = dataset2.flatten_race(x)
    return (x,y)

if __name__=="__main__":
    main()
