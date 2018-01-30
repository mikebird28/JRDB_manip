#-*-coding:utf-8-*-
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from skopt import BayesSearchCV
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Dropout,GaussianNoise
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

CACHE_PATH = "./cache/dnn_regressor2"
MODEL_PATH = "./models/dnn_regressor2"
pd.options.display.max_rows = 1000

def main(use_cache = False):
    predict_type = "win_payoff"
    config = util.get_config("config/config.json")
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
    features = config.features 

    x,y = dataset2.load_dataset(db_con,
        features+["info_race_course_code","rinfo_discipline","rinfo_distance",
                  "pre1_race_course_code","pre1_discipline","pre1_distance",
                  "pre2_race_course_code","pre2_discipline","pre2_distance",
                  "pre3_race_course_code","pre3_discipline","pre3_distance"],
        ["is_win","win_payoff","is_place","place_payoff"])
    con = concat(x,y)
    x_col = x.columns
    y_col = y.columns
    con = con[con["info_race_course_code"] != 0]
    con = con[con["rinfo_discipline"] != 0]
    con = con[con["pre1_discipline"] != 0]
    con = con[con["pre1_distance"] != 0]
    con = con[con["pre1_race_course_code"] != 0]
    con = con[con["pre2_discipline"] != 0]
    con = con[con["pre2_distance"] != 0]
    con = con[con["pre2_race_course_code"] != 0]
    con = con[con["pre3_distance"] != 0]
    con = con[con["pre3_race_course_code"] != 0]
    con.reset_index(drop = True,inplace = True)
 
    x = con.loc[:,x_col]
    y = con.loc[:,y_col]

    p2v_0 = place2vec.get_vector(x["rinfo_discipline"],x["info_race_course_code"],x["rinfo_distance"],prefix = "pre0")
    x = x.drop("info_race_course_code",axis = 1)
    x = x.drop("rinfo_discipline",axis = 1)
    x = x.drop("rinfo_distance",axis = 1)

    p2v_1 = place2vec.get_vector(x["pre1_discipline"],x["pre1_race_course_code"],x["pre1_distance"],prefix = "pre1")
    x = x.drop("pre1_race_course_code",axis = 1)
    x = x.drop("pre1_discipline",axis = 1)
    x = x.drop("pre1_distance",axis = 1)

    p2v_2 = place2vec.get_vector(x["pre2_discipline"],x["pre2_race_course_code"],x["pre2_distance"],prefix = "pre2")
    x = x.drop("pre2_race_course_code",axis = 1)
    x = x.drop("pre2_discipline",axis = 1)
    x = x.drop("pre2_distance",axis = 1)

    p2v_3 = place2vec.get_vector(x["pre3_discipline"],x["pre3_race_course_code"],x["pre3_distance"],prefix = "pre3")
    x = x.drop("pre3_race_course_code",axis = 1)
    x = x.drop("pre3_discipline",axis = 1)
    x = x.drop("pre3_distance",axis = 1)

    col_dic = dataset2.nominal_columns(db_con)
    nom_col = dataset2.dummy_column(x,col_dic)
    x = dataset2.get_dummies(x,col_dic)
    #features = sorted(x.columns.drop(["info_race_id","pre1_distance","pre2_distance","pre3_distance"]).values.tolist())

    x = concat(x,p2v_0)
    x = concat(x,p2v_1)
    x = concat(x,p2v_2)
    x = concat(x,p2v_3)
    x = dataset2.downcast(x)
    features = sorted(x.columns.drop(["info_race_id"]).values.tolist())

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
    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = nom_col)

    print(">> under sampling train dataset")
    train_x.reset_index(inplace = True,drop = True)
    train_y.reset_index(inplace = True,drop = True)
    #train_x,train_y = dataset2.under_sampling(train_x,train_y,key = predict_type,magnif = 2)
    train_x,train_y = dataset2.for_use(train_x,train_y,predict_type)

    sorted_features = sorted(train_x.columns.values.tolist())
    train_x = train_x.loc[:,sorted_features]


    print(">> filling none value of test dataset")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)

    sorted_features = sorted(test_x.columns.values.tolist())
    test_x = test_x.loc[:,sorted_features]

    print(">> under sampling test dataset")
    test_rx,test_ry,test_r_win,test_rp_win,test_r_place,test_rp_place = dataset2.to_races(
        test_x,
        test_y[predict_type],
        test_y["is_win"],
        test_y["win_payoff"],
        test_y["is_place"],
        test_y["place_payoff"]
    )
    #test_x,test_y = dataset2.under_sampling(test_x,test_y,key = predict_type)
    test_x,test_y = dataset2.for_use(test_x,test_y,predict_type)
    test_x = test_x.loc[:,features]

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
    return datasets

def load_datasets_with_p2v():
    pass

def normalize(x,y):
    pass

def create_model(activation = "relu",dropout = 0.8,hidden_1 = 80,hidden_2 =80,hidden_3 = 80):
    #def create_model(activation = "relu",dropout = 0.3,hidden_1 = 200,hidden_2 =250,hidden_3 = 135):
    #Best Paramater of 2 hidden layer : h1 = 50, h2  = 250, dropout = 0.38
    #Best Paramater of 3 hidden layer : h1 = 138, h2  = 265, h3 = 135 dropout = 0.33 
    l2_lambda = 0.0
    hidden_1 = hidden_2 = hidden_3 = 80

    nn = Sequential()
    nn.add(GaussianNoise(0.001,input_shape = (316,)))

    nn.add(Dense(units=hidden_1 , W_regularizer = l2(l2_lambda)))
    nn.add(Activation(activation))
    nn.add(BatchNormalization())
    nn.add(Dropout(dropout))

    nn.add(Dense(units=hidden_2,W_regularizer = l2(l2_lambda)))
    nn.add(Activation(activation))
    nn.add(BatchNormalization())
    nn.add(Dropout(dropout))

    depth = 2
    for i in range(depth):
        nn.add(Dense(units=hidden_3,W_regularizer = l2(l2_lambda)))
        nn.add(Activation(activation))
        nn.add(BatchNormalization())
        nn.add(Dropout(dropout))
    nn.add(Dense(units=1))
    #nn.add(Activation('sigmoid'))
    nn.compile(loss = "binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    return nn

def dnn(features,datasets):
    train_x = np.array(datasets["train_x"])
    train_y = (datasets["train_y"]/100.0).clip(upper = 30.0)
    train_y = np.array(train_y)
    #train_win = np.array(datasets["train_y"])
    test_x  = np.array(datasets["test_x"])
    test_y  = np.array(datasets["test_y"])/100.0
    #test_win  = np.array(datasets["test_y"])
    test_rx = dataset2.races_to_numpy(datasets["test_rx"])
    #test_ry = dataset2.races_to_numpy(datasets["test_ry"])
    test_r_win = dataset2.races_to_numpy(datasets["test_r_win"])
    test_r_place = dataset2.races_to_numpy(datasets["test_r_place"])
    test_rp_win = dataset2.races_to_numpy(datasets["test_rp_win"])
    test_rp_place = dataset2.races_to_numpy(datasets["test_rp_place"])
 
    model = create_model()
    for i in range(1000):
        print(i)
        model.fit(train_x,train_y,epochs = 1,batch_size = 1000)
        score = model.evaluate(test_x,test_y,verbose = 0)

        print("")

        print("test loss : {0}".format(score[0]))
        #print("test acc : {0}".format(score[1]))
        win_eval  = evaluate.top_n_k_keras(model,test_rx,test_r_win,test_rp_win)
        print("[win]   accuracy : {0}, payoff : {1}".format(*win_eval))
        place_eval  = evaluate.top_n_k_keras(model,test_rx,test_r_place,test_rp_place)
        print("[place] accuracy : {0}, payoff : {1}".format(*place_eval))
        save_model(model,MODEL_PATH)

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

"""
def dnn_wigh_bayessearch2(features,datasets):
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

    def data():
        return (train_x,train_y,test_x,test_y)

    def create_model_for_keras(train_x,train_y,test_x,test_y):
        nn = Sequential()
        nn.add(Dense(units={choice,input_dim = 294, activity_regularizer = l2(0.0)))
        nn.add(Activation(activation))
        nn.add(BatchNormalization())
        nn.add(Dropout(dropout))

        nn.add(Dense(units=hidden_2,activity_regularizer = l2(0.0)))
        nn.add(Activation(activation))
        nn.add(BatchNormalization())
        nn.add(Dropout(dropout))

        nn.add(Dense(units=1))
        nn.add(Activation('sigmoid'))
        nn.compile(loss = "mean_squared_loss",optimizer="adam",metrics=["accuracy"])
        #nn.compile(loss = "binary_crossentropy",optimizer="adam",metrics=["accuracy"])
        return nn


    model =  KerasClassifier(create_model,epochs = 6,verbose = 0)
    paramaters = {
        "hidden_1" : (10,500),
        "hidden_2" : (10,500),
        "dropout" : (0.3,0.9),
        "batch_size" : (10,2000),
    }

    cv = BayesSearchCV(model,paramaters,cv = 5,scoring='accuracy',n_iter = 10,verbose = 10)
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
"""

def save_model(model,path):
    print("Save model")
    model.save(path)



def dataset_for_pca(x,y,mean = None,std = None):
    #x = dataset2.normalize(x,mean = mean,std = std)
    x,y = dataset2.pad_race(x,y)
    x = dataset2.flatten_race2(x)
    return (x,y)

def concat(a,b):
    a.reset_index(inplace = True,drop = True)
    b.reset_index(inplace = True,drop = True)
    return pd.concat([a,b],axis = 1)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="horse race result predictor using multilayer perceptron")
    parser.add_argument("-c","--cache",action="store_true",default=False)
    args = parser.parse_args()
    main(use_cache = args.cache)
