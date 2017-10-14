-*- coding:utf-8 -*-

def main():
    config = util.get_config("config/config.json")
    #generate dataset
    db_path = "db/output_v6.db"
    predict_type = "is_win"

    print(">> loading dataset")
    x,y = dataset2.load_dataset(db_path,config.features,["info_race_course_code"])
    col_dic = dataset2.nominal_columns(db_path)
    nom_col = dataset2.dummy_column(x,col_dic)
    x = dataset2.get_dummies(x,col_dic)

    print(">> separating dataset")
    train_x,test_x,train_y,test_y = dataset2.split_with_race(x,y)

    print(">> filling none value of train dataset")
    train_x = dataset2.fillna_mean(train_x,"horse")
    mean = train_x.mean(numeric_only = True)
    std = train_x.std(numeric_only = True).clip(lower = 1e-4)
    train_x = dataset2.normalize(train_x,mean = mean,std = std,remove = nom_col)

    print(">> under sampling train dataset")
    #train_x,train_y = dataset2.for_use(train_x,train_y,predict_type)
    train_x = train_x.drop("info_race_id",axis = 1)
    train_y = train_y.as_matrix()
 

    print(">> filling none value of test dataset")
    test_x = dataset2.fillna_mean(test_x,"horse")
    test_x = dataset2.normalize(test_x,mean = mean,std = std,remove = nom_col)

    print(">> under sampling test dataset")
    #test_x,test_y = dataset2.for_use(test_x,test_y,predict_type)
    test_x = test_x.drop("info_race_id",axis = 1)
    test_y = test_y.as_matrix()
 

    datasets = {
        "train_x"      : train_x,
        "train_y"      : train_y,
        "test_x"       : test_x,
        "test_y"       : test_y,
    }

    #xgbc(train_x.columns,datasets)
    xgbc_wigh_bayessearch(train_x.columns,datasets)
    #xgbc_wigh_gridsearch(train_x.columns,train_x,train_y,test_x,test_y,test_rx,test_ry)

def create_model(activation = "relu",dropout = 0.7,hidden_1 = 100):
    nn = Sequential()
    nn.add(Dense(units=hidden_1))
    nn.add(Activation(activation))
    nn.add(Dense(units=1))
    nn.compile(loss = "binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    return nn

def place2vec(features,datasets):
    train_x = datasets["train_x"]
    train_y = datasets["train_y"]
    test_x = datasets["test_x"]
    test_y = datasets["test_y"]
    
    model =  KerasClassifier(create_model)
    model.fit(train_x,train_y)

    pred = model.predict(test_x)

def horse2vec(features,datasets):
    passc


