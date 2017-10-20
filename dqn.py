
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

LOOSE_VALUE = 0.1
DONT_BUY_VALUE = 0.001


def main():
    predict_type = "win_payoff"
    config = util.get_config("config/config.json")
    db_path = "db/output_v6.db"
    use_cache = False

    datasets = generate_dataset(predict_type,db_path,config)

    dnn(config.features,datasets)
    #dnn_wigh_bayessearch(config.features,datasets)
    #dnn_wigh_gridsearch(train_x.columns,train_x,train_y,test_x,test_y,test_rx,test_ry)

def generate_dataset(predict_type,db_path,config):
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
    train_x = dataset2.fillna_mean(train_x,"horse")
    mean = train_x.mean(numeric_only = True)
    std = train_x.std(numeric_only = True).clip(lower = 1e-4)
    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = nom_col)


    #x = dataset2.normalize(x,mean = mean,std = std)
    print("padding")
    train_x,train_y = dataset2.pad_race(train_x,train_y)
    train_y["dont_buy"] = np.zeros(len(train_y.index),dtype = "float32")-0.01
    print(train_y)
    print("convert_to_race")
    train_x,train_action = dataset2.to_races(train_x,train_y[["win_payoff","dont_buy"]])


    #train_y = pd.DataFrame(train_y,columns = ["buy"])
    print(train_y)


    print(">> under sampling train dataset")
    #train_x,train_y = dataset2.under_sampling(train_x,train_y,key = "is_win")
    train_x,train_y = dataset2.for_use(train_x,train_y,predict_type)

    print(">> filling none value of test dataset")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)

    print(">> under sampling test dataset")
    test_rx,test_ry,test_r_win,test_rp_win,test_r_place,test_rp_place = dataset2.to_races(
        test_x,
        test_y[predict_type],
        test_y["is_win"],
        test_y["win_payoff"],
        test_y["is_place"],
        test_y["place_payoff"]
    )
    #test_x,test_y = dataset2.under_sampling(test_x,test_y,key = "is_win")
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
    return datasets

def create_model(activation = "relu",dropout = 0.8,hidden_1 = 200,hidden_2 = 100):
    nn = Sequential()
    nn.add(Dense(units=hidden_1,input_dim = 172, kernel_initializer = "he_normal"))
    nn.add(Activation(activation))
    nn.add(Dropout(dropout))

    nn.add(Dense(units=hidden_2, kernel_initializer = "he_normal"))
    nn.add(Activation(activation))
    nn.add(Dropout(dropout))

    nn.add(Dense(units=2))
    nn.compile(loss = "mean_squared_error",optimizer="adam",metrics=["accuracy"])
    return nn

def dnn(features,datasets):
    train_x = np.array(datasets["train_x"])
    train_y = np.array(datasets["train_y"])
    test_x  = np.array(datasets["test_x"])
    test_y  = np.array(datasets["test_y"])
    train_rx = datasets["train_rx"]
    train_ry = datasets["train_ry"]
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


if __name__=="__main__":
    main()
