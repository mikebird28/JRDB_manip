# -*- coding:utf-8 -*-
import dataset2
import pandas as pd
import sqlite3
import pickle

KEY_TRAIN_X = "train_x"
KEY_TRAIN_Y = "train_y"
KEY_TRAIN_ODDS = "train_odds"
KEY_TEST_X  = "test_x"
KEY_TEST_y = "test_y"
KEY_TEST_ODDS = "train_odds"
KEY_TEST_RACE_X = "train_rx"
KEY_TEST_RACE_Y = "train_rx"
KEY_TEST_RACE_ODDS = "train_rx"

def load_from_cache(path):
    with open(path) as fp:
        dp = pickle.load(fp)
        return dp
    raise Exception("can not open pickle file")

def load_from_database(path,x_columns,y_columns,odds_columns,where = ""):
    dp = DataProcessor()
    dp.load_from_sql("db/output_v17.db",x_columns,y_columns,odds_columns,where = where)
    return dp

class DataProcessor():
    def __init__(self):
        self.race_mode = False
        self.categorical_columns = []
        self.dummy_columns = []
        self.mean = None
        self.std  = None
        self.train_x = None
        self.train_y = None
        self.train_odds = None
        self.test_x = None
        self.test_y = None
        self.test_odds = None

        self.train_rx = None
        self.train_ry = None
        self.train_odds = None
        self.test_rx = None
        self.test_ry = None
        self.test_rodds = None

    def load_from_sql(self,db_path,x_columns,y_columns,odds_columns,where = "",test_nums = 1000):
        db_con = sqlite3.connect(db_path)
        x_columns = x_columns + ["info_race_id"]
        concat_columns = sorted(x_columns + y_columns + odds_columns)
        df = read_sql(db_con,concat_columns,where)

        use_dummy = True
        if use_dummy:
            pass
        self.mean = df.mean(numeric_only = True)
        self.std = df.std(numeric_only = True).clip(lower = 1e-4)
        self.categorical_columns = dataset2.nominal_columns(db_con)

        train_df,test_df = split_with_race(df)
        self.train_x = train_df.loc[:,x_columns]
        self.train_y = train_df.loc[:,y_columns]
        self.train_odds = train_df.loc[:,odds_columns]
        self.test_x  = test_df.loc[:,x_columns]
        self.test_y = test_df.loc[:,y_columns]
        self.test_odds = test_df.loc[:,odds_columns]

    def dummy(self):
        self.train_x = dataset2.get_dummies(self.train_x,self.categorical_columns)
        self.test_x = dataset2.get_dummies(self.test_x,self.categorical_columns)
        self.dummy_columns = dataset2.dummy_column(self.train_x,self.categorical_columns)
        
    def normalize(self):
        print(">> normalizing datasets")
        self.train_x = dataset2.normalize(self.train_x,mean = self.mean,std = self.std,remove = self.categorical_columns)
        self.test_x = dataset2.normalize(self.test_x,mean = self.mean,std = self.std,remove = self.categorical_columns)

    def standardize(self):
        pass

    def fillna_mean(self):
        print(">> filling none value of datasets")
        self.train_x = fillna_mean(self.train_x,self.mean)
        self.test_x = fillna_mean(self.test_x,self.mean)
        self.train_x = dataset2.downcast(self.train_x)
        self.test_x = dataset2.downcast(self.test_x)

    def to_race_panel(self):
        print(">> converting to race panel")
        if self.race_mode:
            return
        self.train_x,self.train_y,self.train_odds = self.__get_race_panel(self.train_x,self.train_y,self.train_odds)
        self.test_x,self.test_y,self.test_odds = self.__get_race_panel(self.test_x,self.test_y,self.test_odds)
        self.race_mode = True

    def use_race_test_df(self):
        self.test_rx,self.test_ry,self.test_rodds = self.__get_race_panel(self.test_x,self.test_y,self.test_odds)

    def __get_race_panel(self,x,y,odds):
        x,y,odds = dataset2.pad_race(x,y,odds)
        x = dataset2.downcast(x)
        y = dataset2.downcast(y)
        odds = dataset2.downcast(odds)
        x,y,odds = dataset2.to_race_panel(x,y,odds)
        return (x,y,odds)

    def get(self,df_key):
        dic = {
            KEY_TRAIN_X : self.train_x,
            KEY_TRAIN_Y : self.train_y,
            KEY_TRAIN_ODDS : self.train_odds,
            KEY_TEST_X : self.test_x,
            KEY_TEST_y : self.test_y,
            KEY_TEST_ODDS : self.test_odds
        }
        return dic[df_key]

    def save(self,path):
        with open(path,"w") as fp:
            pickle.dump(self,fp)

    def summay(self):
        train_total_size =  len(self.train_x)
        test_total_size =  len(self.test_x)
        print("[*]total train data size : {0}".format(train_total_size))
        print("[*]total test data size  : {0}".format(test_total_size))

def split_with_race(df,test_nums = 1000):
    race_id = pd.DataFrame(df["info_race_id"].unique(),columns = ["info_race_id"])
    race_id["sort_key"] = race_id["info_race_id"].apply(lambda x:x[2:])
    race_id = race_id.sort_values(by = "sort_key")["info_race_id"]
    test_id = race_id[-test_nums:]
    test_df = df[df["info_race_id"].isin(test_id)]
    train_df = df[~df["info_race_id"].isin(test_id)]
    train_df.reset_index(inplace = True,drop = True)
    test_df.reset_index(inplace = True,drop = True)
    return (train_df,test_df)

def read_sql(db_con,columns,where):
    columns_query = ",".join(columns)
    if where.strip() != "":
        sql = "SELECT {0} FROM feature WHERE {1};".format(columns_query,where)
    else:
        sql = "SELECT {0} FROM feature;".format(columns_query)
    df = pd.read_sql(sql,db_con,columns = columns)
    return df

def fillna_mean(df,mean = None):
    if type(mean) == type(None):
        mean = df.mean(numeric_only = True)
        df = df.fillna(mean)
    else:
        df = df.fillna(mean)
    return df

if __name__ == "__main__":
    x_columns = ["addinfo_age_month","info_sex_code"]
    y_columns = ["win_payoff"]
    odds_columns = ["linfo_win_odds","linfo_place_odds"]
    where = "info_year > 12 and info_year < 90"

    #dp = DataProcessor()
    #dp.load_from_sql("db/output_v17.db",x_columns,y_columns,odds_columns,where = where)
    dp = load_from_database("db/output_v17.db",x_columns,y_columns,odds_columns,where = where)
    #dp = load_from_cache("test.pickle")
    dp.dummy()
    dp.fillna_mean()
    dp.normalize()
    #dp.to_race_panel()
    dp.summay()
    dp.save("test.pickle")
    train_x = dp.get(KEY_TRAIN_X)
    print(train_x)


