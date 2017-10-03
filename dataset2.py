import sqlite3
import sys
import pandas as pd
import random
import numpy as np
import feature
import util

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

    #dataset["info_race_id"] = dataset_x["info_race_id"].astype("category")
    #dataset.set_index("info_race_id")
    #dataset.sort_index()

    #dataset_x = dataset[x_col]
    #dataset_y = dataset[y_col]

    return dataset_x,dataset_y

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

def add_race_info(x,race_info):
    x = x.merge(race_info,on = "info_race_id")
    return x

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
        


def get_dummies(x,y):
    pass

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

def normalize(df,typ):
    if typ == "horse":
        return __normalize_horse(dataset)
    elif typ == "race":
        return __normalize_race(dataset)
    else:
        raise Exception("unknown separate type")

def __normalize_horse(dataset):
    pass

def __normalize_race(datast):
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
 

