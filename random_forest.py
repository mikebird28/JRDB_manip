
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import sqlite3
import feature
import reader
import util
import dataset2

def main():
    config = util.get_config("config/config.json")
    #generate dataset
    db_path = "db/output_v5.db"
    pca = PCA(n_components = 15)

    print(">> loading dataset")
    x,y = dataset2.load_dataset(db_path,config.features)
    col_dic = dataset2.nominal_columns(db_path)
    nom_col = dataset2.dummy_column(x,col_dic)
    #x = dataset2.get_dummies(x,col_dic)

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y)

    print(">> filling none value of train dataset")
    #train_x = dataset2.fillna_mean(train_x,"race")
    train_x = dataset2.fillna_mean(train_x,"horse")
    mean = train_x.mean(numeric_only = True)
    std = train_x.std(numeric_only = True).clip(lower = 1e-4)
    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = nom_col)
    #train_x = dataset2.normalize(train_x,typ = "race")

    """
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
    """

    print(">> under sampling train dataset")
    train_x,train_y = dataset2.under_sampling(train_x,train_y)
    train_x,train_y = dataset2.for_use(train_x,train_y)

    print(">> filling none value of test dataset")
    #test_x = dataset2.fillna_mean(test_x,"race")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)

    """
    print(">> generating test pca dataset")
    pca_x,pca_y = dataset_for_pca(test_x,test_y,mean = mean,std = std)
    pca_idx = pca_x["info_race_id"]
    pca_x,pca_y = dataset2.for_use(pca_x,pca_y)
    pca_df = pca.transform(pca_x)
    pca_df = pd.concat([pd.DataFrame(pca_df),pca_idx],axis = 1)
    test_x,test_y = dataset2.add_race_info(test_x,test_y,pca_df)
    """

    print(">> under sampling test dataset")
    test_rx,test_ry = dataset2.to_races(test_x,test_y,to_numpy = True)
    test_x,test_y = dataset2.under_sampling(test_x,test_y)
    test_x,test_y = dataset2.for_use(test_x,test_y)

    random_forest(train_x.columns,train_x,train_y,test_x,test_y,test_rx,test_ry)
    #dnn_wigh_gridsearch(train_x.columns,train_x,train_y,test_x,test_y,test_rx,test_ry)

def random_forest(features,train_x,train_y,test_x,test_y,test_rx,test_ry):
    rfc = RandomForestClassifier(n_estimators = 100)
    rfc.fit(train_x,train_y)
    pred = rfc.predict(test_x)
    accuracy = accuracy_score(test_y,pred)
    top_1_k  = evaluate.top_n_k(rfc,test_rx,test_ry)
    report = classification_report(test_y,pred)
    print("")
    print("Accuracy: {0}".format(accuracy))
    print("top_1_k : {0}".format(top_1_k))
    print(report)
    importances = xgbc.feature_importances_
    evaluate.show_importance(features,importances)


if __name__=="__main__":
    main()
