
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from skopt import BayesSearchCV
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Dropout
from keras.wrappers.scikit_learn import KerasRegressor
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
    db_path = "db/output_v6.db"
    predict_type = "win_payoff"
    pca = PCA(n_components = 100)

    print(">> loading dataset")
    x,y = dataset2.load_dataset(db_path,config.features,["is_win","win_payoff","is_place","place_payoff"])
    y["win_payoff"] = y["win_payoff"].clip(upper = 1000)/1000
    y["place_payoff"] = y["place_payoff"].clip(upper = 1000)/1000
    col_dic = dataset2.nominal_columns(db_path)
    nom_col = dataset2.dummy_column(x,col_dic)
    x = dataset2.get_dummies(x,col_dic)

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y)
    del x
    del y

    print(">> filling none value of train dataset")
    #train_x = dataset2.fillna_mean(train_x,"race")
    train_x = dataset2.fillna_mean(train_x,"horse")
    mean = train_x.mean(numeric_only = True)
    std = train_x.std(numeric_only = True).clip(lower = 1e-4)
    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = nom_col)
    #train_x = dataset2.normalize(train_x,typ = "race")

    """
    print(">> generating pca dataset")
    pca_x,pca_y = dataset_for_pca(train_x,train_y)
    #pca_idx = pca.index
    pca_idx = pca_x["info_race_id"]
    pca_x,pca_y = dataset2.for_use(pca_x,pca_y,predict_type)

    print(">> fitting with pca")
    pca.fit(pca_x)
    print(sum(pca.explained_variance_ratio_))
    print(pca.explained_variance_ratio_)
    pca_df = pd.DataFrame(pca.transform(pca_x).astype(np.float32))
    pca_df = pd.concat([pd.DataFrame(pca_df),pca_idx],axis = 1)
    train_x,train_y = dataset2.add_race_info(train_x,train_y,pca_df)
    """

    print(">> under sampling train dataset")
    train_x,train_y = dataset2.under_sampling(train_x,train_y,key = "is_win")
    train_x,train_y = dataset2.for_use(train_x,train_y,predict_type)

    print(">> filling none value of test dataset")
    #test_x = dataset2.fillna_mean(test_x,"race")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)

    """
    print(">> generating test pca dataset")
    pca_x,pca_y = dataset_for_pca(test_x,test_y,mean = mean,std = std)
    pca_idx = pca_x["info_race_id"]
    pca_x,pca_y = dataset2.for_use(pca_x,pca_y,predict_type)
    pca_df = pca.transform(pca_x).astype(np.float32)
    pca_df = pd.concat([pd.DataFrame(pca_df),pca_idx],axis = 1)
    test_x,test_y = dataset2.add_race_info(test_x,test_y,pca_df)
    """

    print(">> under sampling test dataset")
    test_rx,test_ry,test_r_win,test_rp_win,test_r_place,test_rp_place = dataset2.to_races(
        test_x,
        test_y[predict_type],
        test_y["is_win"],
        test_y["win_payoff"],
        test_y["is_place"],
        test_y["place_payoff"]
    )
    test_x,test_y = dataset2.under_sampling(test_x,test_y,key = "is_win")
    test_x,test_y = dataset2.for_use(test_x,test_y,predict_type)

    datasets = {
        "train_x"      : train_x,
        "train_y"      : train_y,
        "test_x"       : test_x,
        "test_y"       : test_y,
        "test_rx"      : test_rx,
        "test_ry"      : test_ry,
        "test_r_win"   : test_r_win,
        "test_r_place" : test_r_place,
        "test_rp_win"  : test_rp_win,
        "test_rp_place": test_rp_place
    }

    dnn(config.features,datasets)
    #dnn_wigh_bayessearch(config.features,datasets)
    #dnn_wigh_gridsearch(train_x.columns,train_x,train_y,test_x,test_y,test_rx,test_ry)

def create_model(activation = "relu",dropout = 0.8,hidden_1 = 200,hidden_2 = 100):
    nn = Sequential()
    nn.add(Dense(units=hidden_1,input_dim = 172, kernel_initializer = "he_normal"))
    nn.add(Activation(activation))
    nn.add(Dropout(dropout))

    nn.add(Dense(units=hidden_2, kernel_initializer = "he_normal"))
    nn.add(Activation(activation))
    nn.add(Dropout(dropout))

    nn.add(Dense(units=1))
    nn.compile(loss = "mean_squared_error",optimizer="adam",metrics=["accuracy"])
    return nn

def dnn(features,datasets):
    train_x = np.array(datasets["train_x"])
    train_y = np.array(datasets["train_y"])
    test_x  = np.array(datasets["test_x"])
    test_y  = np.array(datasets["test_y"])
    test_rx = dataset2.races_to_numpy(datasets["test_rx"])
    test_ry = dataset2.races_to_numpy(datasets["test_ry"])
    test_r_win = dataset2.races_to_numpy(datasets["test_r_win"])
    test_r_place = dataset2.races_to_numpy(datasets["test_r_place"])
    test_rp_win = dataset2.races_to_numpy(datasets["test_rp_win"])
    test_rp_place = dataset2.races_to_numpy(datasets["test_rp_place"])
 
    model =  KerasRegressor(create_model,batch_size = 300,verbose = 1,epochs = 15)
    model.fit(train_x,train_y)

    pred = model.predict(test_x)

    win_eval  = evaluate.top_n_k_regress(model,test_rx,test_r_win,test_rp_win)
    print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
    place_eval  = evaluate.top_n_k_regress(model,test_rx,test_r_place,test_rp_place)
    print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))

    #report = classification_report(test_y,pred)
    #print(report)


def dnn_wigh_gridsearch(features,datasets):
    train_x = datasets["train_x"]
    train_y = datasets["train_y"]
    test_x  = datasets["test_x"]
    test_y  = datasets["test_y"]
    test_rx = datasets["test_rx"]
    test_ry = datasets["test_ry"]
    test_r_win = datasets["test_r_win"]
    test_r_place = datasets["test_r_place"]
    test_rp_win = datasets["test_rp_win"]
    test_rp_place = datasets["test_rp_place"]
 
    model =  KerasRegressor(create_model,nb_epoch = 6)
    paramaters = {
        hidden_1 : [50,100],
        hidden_2 : [50],
        dropout  : [1.0],
    }

    cv = GridSearchCV(model,paramaters,cv = 2,scoring='accuracy',verbose = 2)
    cv.fit(train_x,train_y)

    pred = cv.predict(test_x)

    win_eval  = evaluate.top_n_k(cv,test_rx,test_r_win,test_rp_win)
    print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
    place_eval  = evaluate.top_n_k(cv,test_rx,test_r_place,test_rp_place)
    print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))

    #report = classification_report(test_y,pred)
    #print(report)

    print("Paramaters")
    best_parameters = cv.best_params_
    for pname in sorted(best_parameters.keys()):
        print("{0} : {1}".format(pname,best_parameters[pname]))

def dnn_wigh_bayessearch(features,datasets):
    train_x = np.array(datasets["train_x"])
    train_y = np.array(datasets["train_y"])
    test_x  = np.array(datasets["test_x"])
    test_y  = np.array(datasets["test_y"])
    test_rx = dataset2.races_to_numpy(datasets["test_rx"])
    test_ry = dataset2.races_to_numpy(datasets["test_ry"])
    test_r_win = dataset2.races_to_numpy(datasets["test_r_win"])
    test_r_place = dataset2.races_to_numpy(datasets["test_r_place"])
    test_rp_win = dataset2.races_to_numpy(datasets["test_rp_win"])
    test_rp_place = dataset2.races_to_numpy(datasets["test_rp_place"])
 
    model =  KerasClassifier(create_model,epochs = 6,verbose = 0)
    paramaters = {
        "hidden_1" : (50,250),
        "hidden_2" : (50,250),
        "dropout" : (0.3,1.0)
    }

    cv = BayesSearchCV(model,paramaters,cv = 3,scoring='accuracy',n_iter = 10,verbose = 2)
    cv.fit(train_x,train_y)

    pred = cv.predict(test_x)
    accuracy = accuracy_score(test_y,pred)
    print("")
    print("Accuracy: {0}".format(accuracy))

    win_eval  = evaluate.top_n_k(cv,test_rx,test_r_win,test_rp_win)
    print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
    place_eval  = evaluate.top_n_k(cv,test_rx,test_r_place,test_rp_place)
    print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))

    report = classification_report(test_y,pred)
    print(report)

    print("Paramaters")
    best_parameters = cv.best_params_
    for pname in sorted(best_parameters.keys()):
        print("{0} : {1}".format(pname,best_parameters[pname]))


def dataset_for_pca(x,y,mean = None,std = None):
    #x = dataset2.normalize(x,mean = mean,std = std)
    x,y = dataset2.pad_race(x,y)
    x = dataset2.flatten_race2(x)
    return (x,y)

def dnn_with_keras(f,train_x,train_y,test_x,test_y,test_rx,test_ry):
    pass

if __name__=="__main__":
    main()
