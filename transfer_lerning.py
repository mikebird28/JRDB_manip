
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
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
    db_path = "db/output_v6.db"
    pca = PCA(n_components = 15)

    print(">> loading dataset")
    x,y = dataset2.load_dataset(db_path,config.features,"win")
    col_dic = dataset2.nominal_columns(db_path)
    nom_col = dataset2.dummy_column(x,col_dic)
    x = dataset2.get_dummies(x,col_dic)

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y)

    print(">> filling none value of train dataset")
    #train_x = dataset2.fillna_mean(train_x,"race")
    train_x = dataset2.fillna_mean(train_x,"horse")
    mean = train_x.mean(numeric_only = True)
    std = train_x.std(numeric_only = True).clip(lower = 1e-4)
    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = nom_col)
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
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)

    print(">> generating test pca dataset")
    pca_x,pca_y = dataset_for_pca(test_x,test_y,mean = mean,std = std)
    pca_idx = pca_x["info_race_id"]
    pca_x,pca_y = dataset2.for_use(pca_x,pca_y)
    pca_df = pca.transform(pca_x)
    pca_df = pd.concat([pd.DataFrame(pca_df),pca_idx],axis = 1)
    test_x,test_y = dataset2.add_race_info(test_x,test_y,pca_df)

    print(">> under sampling test dataset")
    test_rx,test_ry = dataset2.to_races(test_x,test_y,to_numpy = True)
    test_x,test_y = dataset2.under_sampling(test_x,test_y)
    test_x,test_y = dataset2.for_use(test_x,test_y)

    dnn(train_x.columns,train_x,train_y,test_x,test_y,test_rx,test_ry)
    #dnn_wigh_gridsearch(train_x.columns,train_x,train_y,test_x,test_y,test_rx,test_ry)


def keras_to_scikit(model,params):
    def model_func():
        return model
    return KerasClassifier(model_func,epochs = 20,batch_size = 300)

def dnn(features,train_x,train_y,test_x,test_y,test_rx,test_ry):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    nn = Sequential()
    nn.add(Dense(units=150, input_dim=148,name = "feature_layer"))
    nn.add(Activation('relu'))
    nn.add(Dropout(0.7))

    nn.add(Dense(units=50))
    nn.add(Activation('relu'))
    nn.add(Dropout(0.7))

    nn.add(Dense(units=1))
    nn.add(Activation('sigmoid'))
    nn.compile(loss = "binary_crossentropy",optimizer="adam",metrics=["accuracy"])

    model = keras_to_scikit(nn,params = {})
    model.fit(train_x,train_y)

    pred = model.predict(test_x)
    accuracy = accuracy_score(test_y,pred)

    top_1_k  = evaluate.top_n_k(model,test_rx,test_ry)
    report = classification_report(test_y,pred)
    print("")
    print("Accuracy: {0}".format(accuracy))
    print("top_1_k : {0}".format(top_1_k))
    print(report)

    internal_layer = Model(inputs = nn.input,output = nn.get_layer("feature_layer").output)
    train_f = internal_layer.predict(train_x)

    #create internal feature
    test_f = internal_layer.predict(test_x)
    print(test_f)
    test_rf = []
    for r in test_rx:
        rf = internal_layer.predict(r)
        test_rf.append(rf)

    #fit wight xgbc and predict
    xgbc = xgb.XGBClassifier()
    xgbc.fit(train_f,train_y)

    pred = xgbc.predict(test_f)
    report = classification_report(test_y,pred)
    accuracy = accuracy_score(test_y,pred)
    top_1_k  = evaluate.top_n_k(xgbc,test_rf,test_ry)

    print("")
    print("Accuracy: {0}".format(accuracy))
    print("top_1_k : {0}".format(top_1_k))
    print(report)



def dnn_wigh_gridsearch(features,train_x,train_y,test_x,test_y,test_rx,test_ry):
    paramaters = [
    ]
    model = Sequential()
    model.add(Dense(units=50, input_dim=148))
    model.add(Activation('relu'))
    #model.add(Dropout(0.8))
    model.add(Dense(units=50))
    model.add(Activation('relu'))
    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))
    model =  KerasClassifier(build_fn = model, verbose=1,batch_size = 500)

    cv = GridSearchCV(model,paramaters,cv = 2,scoring='accuracy',verbose = 2)
    cv.fit(train_x,train_y)
    pred = cv.predict(test_x)

    accuracy = accuracy_score(test_y,pred)
    top_1_k  = evaluate.top_n_k(cv,test_rx,test_ry)
    report = classification_report(test_y,pred)
    print("")
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

def dnn_with_keras(f,train_x,train_y,test_x,test_y,test_rx,test_ry):
    pass

if __name__=="__main__":
    main()
