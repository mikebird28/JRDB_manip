#-*-coding:utf-8-*-
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from skopt import BayesSearchCV
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Dropout,Input
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l1,l2
import argparse
import numpy as np
import pandas as pd
import sqlite3
import feature
import dataset2
import util
import evaluate
import course2vec
import place2vec
import field_fitness

CACHE_PATH = "./cache/lstm_classify"
pd.options.display.max_rows = 1000

def main(use_cache = False):
    predict_type = "is_win"
    config = util.get_config("config/config_lstm.json")
    db_path = "db/output_v12.db"
    db_con = sqlite3.connect(db_path)

    if use_cache:
        print("[*] load dataset from cache")
        datasets = dataset2.load_cache(CACHE_PATH)
    else:
        datasets = generate_dataset(predict_type,db_con,config)
        dataset2.save_cache(datasets,CACHE_PATH)
    dnn(config.features, datasets)
    #dnn_wigh_bayessearch(config.features,datasets)
    #dnn_wigh_gridsearch(config.features,datasets)


def generate_dataset(predict_type,db_con,config):

    print(">> loading dataset")
    extend_features = []

    features = config.features 
    extend_features.extend(features)

    pre_features = []
    for i in range(5):
        idx = i+1
        pre_i_features = ["pre{0}_{1}".format(idx,f) for f in config.previous]
        pre_features.append(pre_i_features)
        extend_features.extend(pre_i_features)
    target_features = ["is_win","win_payoff","is_place","place_payoff"]

    x,y = dataset2.load_dataset(db_con,extend_features,target_features)
    con = concat(x,y)
    x_col = x.columns
    y_col = y.columns
    x = con.loc[:,x_col]
    y = con.loc[:,y_col]

    """
    col_dic = dataset2.nominal_columns(db_con)
    nom_col = dataset2.dummy_column(x,col_dic)
    x = dataset2.get_dummies(x,col_dic)
    features = sorted(x.columns.drop(["info_race_id"]).values.tolist())
    """
    x = dataset2.downcast(x)

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y)
    train_x = dataset2.downcast(train_x)
    test_x = dataset2.downcast(test_x)
    del x
    del y

    print(">> filling none value of train dataset")
    train_x = dataset2.fillna_mean(train_x,"horse")
    mean = train_x.mean(numeric_only = True)
    std = train_x.std(numeric_only = True).clip(lower = 1e-4)
    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = [])
    #train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = nom_col)

    print(">> under sampling train dataset")
    train_x.reset_index(inplace = True,drop = True)
    train_y.reset_index(inplace = True,drop = True)
    train_x,train_y = dataset2.under_sampling(train_x,train_y,key = predict_type,magnif = 1)

    train_ci  = train_x.loc[:,features]
    train_pi_dic = {}
    for i in range(5):
        f = pre_features[i]
        train_pi_dic[i] = train_x.loc[:,f]
    train_pi = pd.Panel(train_pi_dic)
    train_pi = train_pi.swapaxes(0,1,copy = False)
    del train_pi_dic
    print(train_pi)

    print(">> under sampling train dataset")
    train_x.reset_index(inplace = True,drop = True)
    train_y.reset_index(inplace = True,drop = True)
    train_x,train_y = dataset2.under_sampling(train_x,train_y,key = predict_type,magnif = 1)
    train_x,train_y = dataset2.for_use(train_x,train_y,predict_type)
    train_ci,train_pi,train_y = for_use(train_ci,train_pi,train_y,predict_type)

    print(">> filling none value of test dataset")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = [])
    #test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)

    test_x,test_y = dataset2.under_sampling(test_x,test_y,key = predict_type)
    test_ci  = test_x.loc[:,features]
    test_pi_dic = {}
    for i in range(5):
        f = pre_features[i]
        test_pi_dic[i] = test_x.loc[:,f]
    test_pi = pd.Panel(test_pi_dic)
    test_pi = test_pi.swapaxes(0,1,copy = False)
    del test_pi_dic
    print(test_pi)
    test_ci,test_pi,test_y = for_use(test_ci,test_pi,test_y,predict_type)

    print(">> under sampling test dataset")
    """
    test_rci,test_rpi,test_ry,test_r_win,test_rp_win,test_r_place,test_rp_place = dataset2.to_races(
        test_ci,
        test_pi,
        test_y[predict_type],
        test_y["is_win"],
        test_y["win_payoff"],
        test_y["is_place"],
        test_y["place_payoff"]
    )
    """

    datasets = {
        "train_ci"      : train_ci,
        "train_pi"      : train_ci,
        "train_y"      : train_y,
        "test_ci"       : test_ci,
        "test_pi"       : test_pi,
        "test_y"       : test_y,
        #"test_rci"      : test_rci,
        #"test_rpi"      : test_rpi,
        #"test_ry"      : test_ry,
        #"test_r_win"   : test_r_win,
        #"test_r_place" : test_r_place,
        #"test_rp_win"  : test_rp_win,
        #"test_rp_place": test_rp_place
    }
    return datasets

def create_model(activation = "relu",dropout = 0.3,hidden_1 = 80,hidden_2 =80,hidden_3 = 80):
    #def create_model(activation = "relu",dropout = 0.3,hidden_1 = 200,hidden_2 =250,hidden_3 = 135):
    #Best Paramater of 2 hidden layer : h1 = 50, h2  = 250, dropout = 0.38
    #Best Paramater of 3 hidden layer : h1 = 138, h2  = 265, h3 = 135 dropout = 0.33 
    x = Input(shape = (266,))
    #x = GaussianNoise(0.01)(x)
    #x = Reshape([18,feature_size,1],input_shape = (feature_size*18,))(x)

    x = Dense(units=hidden_1,input_dim = 266, activity_regularizer = l2(0.0))(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(units=hidden_2,activity_regularizer = l2(0.0))(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(units=1)(x)
    x = Activation('sigmoid')
    model = x.compile(loss = "binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    return model

def dnn(features,datasets):
    train_ci = np.array(datasets["train_ci"])
    train_pi = np.array(datasets["train_pi"])
    train_y = np.array(datasets["train_y"])
    test_ci  = np.array(datasets["test_ci"])
    test_pi  = np.array(datasets["test_pi"])
    test_y  = np.array(datasets["test_y"])
    """
    test_rx = dataset2.races_to_numpy(datasets["test_rx"])
    #test_ry = dataset2.races_to_numpy(datasets["test_ry"])
    test_r_win = dataset2.races_to_numpy(datasets["test_r_win"])
    test_r_place = dataset2.races_to_numpy(datasets["test_r_place"])
    test_rp_win = dataset2.races_to_numpy(datasets["test_rp_win"])
    test_rp_place = dataset2.races_to_numpy(datasets["test_rp_place"])
    """
 
    model = create_model()
    for i in range(1000):
        print(i)
        model.fit(train_x,train_y,epochs = 1,batch_size = 1000)
        score = model.evaluate(test_x,test_y,verbose = 0)

        print("")

        print("test loss : {0}".format(score[0]))
        print("test acc : {0}".format(score[1]))
        win_eval  = evaluate.top_n_k_keras(model,test_rx,test_r_win,test_rp_win)
        print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
        place_eval  = evaluate.top_n_k_keras(model,test_rx,test_r_place,test_rp_place)
        print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))

def dnn_wigh_gridsearch(features,datasets):
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
 
    model =  KerasClassifier(create_model,nb_epoch = 6,batch_size = 500)
    paramaters = {
        "hidden_1" : [50,100,200,300],
        "hidden_2" : [50,100,200,300],
        "dropout"  : [0.3,0.4,0.5],
    }

    cv = GridSearchCV(model,paramaters,cv = 3,scoring='accuracy',verbose = 3)
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
        "hidden_1" : (10,500),
        "hidden_2" : (10,500),
        "dropout" : (0.3,0.9),
        "batch_size" : (10,2000),
    }

    cv = BayesSearchCV(model,paramaters,cv = 5,n_iter = 10,verbose = 3)
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

def concat(a,b):
    a.reset_index(inplace = True,drop = True)
    b.reset_index(inplace = True,drop = True)
    return pd.concat([a,b],axis = 1)

def for_use(x1,x2,y,target):
    if "info_race_id" in x1.columns:
        x1 = x1.drop("info_race_id",axis = 1)
    print(x2.axes[2])
    if "info_race_id" in x2.axes[2]:
        x2 = x2.drop("info_race_id",axis = 1)
    #y = y[target].values.tolist()
    y = y.loc[:,target]
    return (x1,x2,y)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
