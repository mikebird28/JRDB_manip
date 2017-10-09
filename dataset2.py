import sqlite3
import sys
import pandas as pd
import random
import numpy as np
import feature
import util
import reader
from sklearn.preprocessing import OneHotEncoder

def load_dataset(db_path,features,typ):
    x_col = ["info_race_id"] + features
    y_col = ["is_win"]
    dataset_x = []
    dataset_y = []

    db_con = sqlite3.connect(db_path)
    f_orm = feature.Feature(db_con)
    for x,y in f_orm.fetch_horse(x_col,typ):
        dataset_x.append(x)
        dataset_y.append(y)
    dataset_x = pd.DataFrame(dataset_x,columns = x_col)
    dataset_y = pd.DataFrame(dataset_y,columns = y_col)
    dataset = pd.concat([dataset_x,dataset_y],axis = 1)

    for col in dataset.columns:
        if col == "info_race_id":
            continue
        #dataset[col] = dataset[col].astype(np.int64)
        dataset[col] = dataset[col].convert_objects(convert_numeric = True)
        dataset[col] = pd.to_numeric(dataset[col],errors = "coerce")

    #dataset["info_race_id"] = dataset_x["info_race_id"].astype("category")
    #dataset.set_index("info_race_id")
    #dataset.sort_index()

    #dataset_x = dataset[x_col]
    #dataset_y = dataset[y_col]

    return dataset_x,dataset_y

def nominal_columns(db_path):
    con = sqlite3.connect(db_path)
    orm = reader.ColumnInfoORM(con)
    dic = orm.column_dict("feature")
    new_dic = {}
    for k,v in dic.items():
        if v.typ == util.NOM_SYNBOL:
            new_dic[k] = v.n
    return new_dic

def split_with_race(x,y):
    x_col = x.columns
    y_col = y.columns

    con = pd.concat([y,x],axis = 1)
    race_id = con["info_race_id"].unique()
    test_id = random.sample(race_id,1000)
    test_con = con[con["info_race_id"].isin(test_id)]
    test_x = test_con.loc[:,x_col]
    test_y = test_con.loc[:,y_col]
    train_con = con[~con["info_race_id"].isin(test_id)]
    train_x = train_con.loc[:,x_col]
    train_y = train_con.loc[:,y_col]
    return train_x,test_x,train_y,test_y

def add_race_info(x,y,race_info):
    x_col = pd.Index(x.columns.tolist() + race_info.columns.tolist()).unique()
    y_col = y.columns
    con = pd.concat([x,y],axis = 1)
    con = con.merge(race_info,on = "info_race_id",how = "left")
    con = con.loc[:,~con.columns.duplicated()]
    ret_x = con.loc[:,x_col]
    ret_y = con.loc[:,y_col]
    return ret_x,ret_y

def pad_race(x,y,n=18):
    x_col = x.columns
    y_col = y.columns
    columns = x_col + y_col
    df = pd.concat([y,x],axis = 1)
    df = df.sort_values(by = "info_race_id",ascending = True)

    size = df.groupby("info_race_id").size().reset_index(name = "counts")
    mean = df.groupby("info_race_id").mean().reset_index()
    mean.loc[:,y.columns] = 0
    target_columns = x_col

    merged = mean.merge(size,on = "info_race_id",how="inner")
    ls = []
    for i,row in merged.iterrows():
        rid = row["info_race_id"]
        hnum = row["counts"]
        pad_num = n - row["counts"]
        new_row = row[target_columns]
        for i in range(pad_num):
            ls.append(new_row)
    pad_data = pd.DataFrame(ls,columns = target_columns)
    df = df.append(pad_data)
    df = df.sort_values(by = "info_race_id")
    size = df.groupby("info_race_id").size().reset_index(name="counts")
    dfx = df.loc[:,x_col]
    dfy = df.loc[:,y_col]
    return (dfx,dfy)

def flatten_race(df):
    ls = []
    rids = []
    count = 0
    x_col = df.columns
    x_col = x_col.drop("info_race_id")
    groups = df.groupby("info_race_id")
    for rid ,g in groups:
        count += 1
        if count%100 == 0:
            print(count)
        flatten = g[x_col].values[0:18].flatten().tolist()
        #flatten = g[x_col].values.flatten().tolist()
        ls.append(flatten)
        rids.append(rid)
    dfx = pd.DataFrame(ls)
    dfid = pd.DataFrame(rids,columns=["info_race_id"])
    df = pd.concat([dfx,dfid],axis = 1)
    return df

def get_dummies(x,col_dic):
    pairs = {}
    for col in x.columns:
        try:
            pairs[col] = col_dic[col] + 1
        except KeyError:
            pass
    pairs = pairs.items()
    if len(pairs) == 0:
        return x
    pairs = sorted(pairs)
    columns = map(lambda x:x[0], pairs)
    n_values = map(lambda x:x[1], pairs)
    column_name = []
    for k,v in pairs:
        cols = ["{0}_{1}".format(k,i) for i in range(v)]
        column_name.extend(cols)
    print(column_name)

    ohe = OneHotEncoder(n_values = n_values,sparse = False)
    x.loc[:,columns] = x.loc[:,columns].fillna(0)
    tmp_x = x.loc[:,columns]

    ohe.fit(tmp_x)
    dummies = ohe.transform(tmp_x)
    dummies = pd.DataFrame(dummies,index = tmp_x.index,columns = column_name)
    x = x.drop(columns,axis = 1)
    x = x.merge(dummies,left_index = True,right_index = True)
    return x

def dummy_column(x,col_dic):
    pairs = {}
    for col in x.columns:
        try:
            pairs[col] = col_dic[col] + 1
        except KeyError:
            pass
    pairs = pairs.items()
    if len(pairs) == 0:
        return x
    pairs = sorted(pairs)
    columns = map(lambda x:x[0], pairs)
    n_values = map(lambda x:x[1], pairs)
    column_name = []
    for k,v in pairs:
        cols = ["{0}_{1}".format(k,i) for i in range(v)]
        column_name.extend(cols)
    return column_name

def fillna_mean(dataset,typ = "horse"):
    if typ == "horse":
        return __fillna_mean_horse(dataset)
    elif typ == "race":
        return __fillna_mean_race(dataset)
    else:
        raise Exception("unknown separate type")

def __fillna_mean_race(df):
    means_stds = df.groupby("info_race_id").agg(["mean"]).reset_index()
    columns = df.columns
    for col in columns:
        if col is "info_race_id":
            continue
        ms = means_stds[col]
        df = df.join(ms,on = "info_race_id")
        df = df.fillna(df["mean"])
        df = df.drop("mean",axis=1)
    return df

def __fillna_mean_horse(df):
    mean = df.mean(numeric_only = True)
    df = df.fillna(mean)
    return df

def fillna_zero(df):
    columns = df.columns
    for col in columns:
        try:
            df = df.fillna(0)
        except Exception:
            pass
    return df

def normalize(dataset,typ="horse",mean = None,std = None,remove = []):
    if typ == "horse":
        return __normalize_horse(dataset,mean,std,remove)
    elif typ == "race":
        return __normalize_race(dataset)
    else:
        raise Exception("unknown separate type")

def __normalize_horse(dataset,mean = None,std = None,remove = []):
    if type(mean) == type(None):
        mean = dataset.mean(numeric_only = True)
    if type(std) == type(None):
        std = dataset.std(numeric_only = True).clip(1e-3)
    dataset = dataset.copy()
    for col in mean.index:
        if col in remove:
            continue
        m = mean[col]
        s = std[col]
        dataset[col] = (dataset[col] - m)/s
    return dataset

def __normalize_race(dataset,mean = None,std = None):
    pass

def under_sampling(x,y):
    x_col = x.columns
    y_col = y.columns
    con = pd.concat([y,x],axis = 1)
    lowest_frequent_value = 1
    low_frequent_records = con.ix[con.iloc[:,0] == lowest_frequent_value,:]
    other_records = con.ix[con.iloc[:,0] != lowest_frequent_value,:]
    under_sampled_records = other_records.sample(len(low_frequent_records))
    con = pd.concat([low_frequent_records,under_sampled_records])
    con.sample(frac=1.0).reset_index(drop=True)
    con_x = con.loc[:,x_col]
    con_y = con.loc[:,y_col]
    return con_x,con_y

def over_sampling(x,y):
    con = pd.concat([y,x],axis = 1)
    highest_frequent_value = 1
    high_frequent_records = con.ix[con.iloc[:,0] == highest_frequent_value,:]
    other_records = con.ix[con.iloc[:,0] != highest_frequent_value,:]
    under_sampled_records = other_records.sample(len(low_frequent_records))
    con = pd.concat([low_frequent_records,under_sampled_records])
    con.sample(frac=1.0).reset_index(drop=True)
    con_x = con.iloc[:,1:]
    con_y = con.iloc[:,0]
    return con_x,con_y

def for_use(x,y):
    x = x.drop("info_race_id",axis = 1)
    y = y["is_win"].values.tolist()
    return (x,y)

def to_races(x,y):
    x["info_race_id"] = x["info_race_id"].astype(str)
    x_col = x.columns
    y_col = y.columns
    races_x = []
    races_y = []
    con = pd.concat([x,y],axis = 1)
    grouped = con.groupby("info_race_id")
    for name,group in grouped:
        rx = group[x_col]
        rx = rx.drop("info_race_id",axis = 1)
        ry = group[y_col].values.tolist()
        races_x.append(rx)
        races_y.append(ry)
    return (races_x,races_y)

if __name__=="__main__":
    config = util.get_config("config/config.json")
    print("loading data")
    dx,dy = load_dataset("db/output_v4.db",config.features,typ="win")
    print("fill with horse mean")
    dx,dy = pad_race(dx,dy)
    dx = flatten_race(dx)
    #fillna_mean(dx,"race")
