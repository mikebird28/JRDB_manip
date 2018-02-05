# -*- coding:utf-8 -*-
import dataset2
import pandas as pd
import sqlite3,pickle

KEY_TRAIN_X = "train_x"
KEY_TRAIN_Y = "train_y"
KEY_TRAIN_ODDS = "train_odds"
KEY_TRAIN_RACE = "train_race"
KEY_TEST_X  = "test_x"
KEY_TEST_Y = "test_y"
KEY_TEST_ODDS = "train_odds"
KEY_TEST_RACE = "test_race"

def load_from_cache(path):
    with open(path) as fp:
        dp = pickle.load(fp)
        return dp
    raise Exception("can not open pickle file")

def load_from_database(path,x_columns,y_columns,odds_columns,where = ""):
    dp = DataProcessor()
    dp.load_from_sql(path,x_columns,y_columns,odds_columns,where = where)
    return dp

class DataProcessor():
    def __init__(self):
        self.race_mode = False
        self.is_dummied = False
        self.categorical_columns = {}
        self.dummy_columns = []
        self.mean = None
        self.std  = None
        self.train_x = None
        self.train_y = None
        self.train_odds = None
        self.test_x = None
        self.test_y = None
        self.test_odds = None

        self.train_race_df = None
        self.test_race_df = None

    def load_from_sql(self,db_path,x_columns,y_columns,odds_columns,where = "",test_nums = 1000):
        print(">> loading datasets from {0}".format(db_path))
        db_con = sqlite3.connect(db_path)
        x_columns = x_columns + ["info_race_id"]
        concat_columns = sorted(x_columns + y_columns + odds_columns)
        df = read_sql(db_con,concat_columns,where)

        use_dummy = True
        if use_dummy:
            pass
        self.mean = df.mean(numeric_only = True)
        self.std = df.std(numeric_only = True).clip(lower = 1e-4)
        self.max = df.max(numeric_only = True)
        self.min = df.min(numeric_only = True)
        self.categorical_columns = dataset2.nominal_columns(db_con)

        train_df,test_df = split_with_race(df)
        self.train_x = train_df.loc[:,x_columns]
        self.train_y = train_df.loc[:,y_columns]
        self.train_odds = train_df.loc[:,odds_columns]
        self.test_x  = test_df.loc[:,x_columns]
        self.test_y = test_df.loc[:,y_columns]
        self.test_odds = test_df.loc[:,odds_columns]
        check_duplicates(self.train_x.columns.values.tolist())

    def dummy(self):
        print(">> adding dummy variables to datasets")
        self.train_x = dataset2.get_dummies(self.train_x,self.categorical_columns)
        self.test_x = dataset2.get_dummies(self.test_x,self.categorical_columns)
        self.dummy_columns = dataset2.dummy_column(self.train_x,self.categorical_columns)
        self.is_dummied = True
        
    def normalize(self,test = False, odds = False):
        if self.is_dummied:
            categorical = self.dummy_columns
        else:
            categorical = self.categorical_columns
        print(">> normalizing datasets")
        self.train_x = dataset2.normalize(self.train_x,mean = self.mean,std = self.std,remove = categorical)
        self.test_x = dataset2.normalize(self.test_x,mean = self.mean,std = self.std,remove = categorical)
        if odds:
            self.train_odds = dataset2.normalize(self.train_odds,mean = self.mean,std = self.std,remove = categorical)
            self.test_odds = dataset2.normalize(self.test_odds,mean = self.mean,std = self.std,remove = categorical)

    def standardize(self):
        print(">> standardize datasets")
        self.train_x = dataset2.standardize(self.train_x, mx = self.max, mn = self.min, remove = self.categorical_dic)
        self.test_x = dataset2.standardize(self.test_x, mx = self.max, mn = self.min, remove = self.categorical_dic)

    def under_sampling(self,key,train_magnif = 1):
        print(">> undersampling datasets ")
        if self.race_mode:
            raise Exception("under_sampling cannot use on race mode")
        self.train_x,self.train_y = dataset2.under_sampling(self.train_x,self.train_y,key = key,magnif = train_magnif)
        self.test_x,self.test_y = dataset2.under_sampling(self.test_x,self.test_y,key = key)

    def fillna_mean(self,typ = "horse"):
        print(">> filling none value of datasets")
        if typ == "horse":
            self.train_x = fillna_mean_horse(self.train_x,self.mean,categorical = self.categorical_columns)
            self.test_x = fillna_mean_horse(self.test_x,self.mean,categorical = self.categorical_columns)
        elif typ == "race":
            self.train_x = fillna_mean_race(self.train_x,categorical = self.categorical_columns)
            self.test_x = fillna_mean_race(self.test_x,categorical = self.categorical_columns)
        self.train_x = dataset2.downcast(self.train_x)
        self.test_x = dataset2.downcast(self.test_x)

    def to_race_panel(self):
        print(">> converting to race panel")
        if self.race_mode:
            return
        self.train_x,self.train_y,self.train_odds = self.__get_race_panel(self.train_x,self.train_y,self.train_odds)
        self.test_x,self.test_y,self.test_odds = self.__get_race_panel(self.test_x,self.test_y,self.test_odds)
        self.race_mode = True

    def keep_separate_race_df(self,test_only = True):
        print(">> keeping race dataframe separete from main data ")
        if self.race_mode:
            raise Exception("datasets were alyready converted to race dataframe. you don't have to keep race datasets separate")
        #self.test_rx,self.test_ry,self.test_rodds = self.__get_race_panel(self.test_x,self.test_y,self.test_odds)
        labels = ["x","odds"] + self.test_y.columns.values.tolist()
        values = [self.test_x,self.test_odds] + [self.test_y.loc[:,lab] for lab in self.test_y.columns]
        outputs = dataset2.to_races(*values)
        self.test_race_df = {}
        for k,v in zip(labels,outputs):
            self.test_race_df[k] = v

        #if not test_only:
        #    self.train_rx,self.train_ry,self.train_rodds = self.__get_race_panel(self.train_x,self.train_y,self.train_odds)

    def __get_race_panel(self,x,y,odds):
        x,y,odds = dataset2.pad_race(x,y,odds)
        x = dataset2.downcast(x)
        y = dataset2.downcast(y)
        odds = dataset2.downcast(odds)
        return dataset2.to_race_panel(x,y,odds)

    def __reset_index(self):
        self.train_x.reset_index(inplace = True)
        self.train_y.reset_index(inplace = True)

    def get(self,df_key,remove_race_id = True):
        dic = {
            KEY_TRAIN_X : self.train_x,
            KEY_TRAIN_Y : self.train_y,
            KEY_TRAIN_ODDS : self.train_odds,
            KEY_TEST_X : self.test_x,
            KEY_TEST_Y : self.test_y,
            KEY_TEST_ODDS : self.test_odds,
            KEY_TRAIN_RACE : self.train_race_df,
            KEY_TEST_RACE : self.test_race_df,
        }
        ret = dic[df_key]
        if remove_race_id and type(ret) == pd.DataFrame and "info_race_id" in ret.columns:
            ret = ret.drop("info_race_id",axis = 1)
        if remove_race_id and type(ret) == pd.Panel and "info_race_id" in ret.axes[2]:
            ret = ret.drop("info_race_id",axis = 2)
        if ret is None:
            error_msg = "{0} is not initialized".format(df_key)
            raise Exception(error_msg)
        return ret

    def save(self,path):
        """
        gb = pow(2,30)
        print(gb)
        for var_name in self.__dict__.keys():
            var = self.__dict__[var_name]
            divide = sys.getsizeof // gb
            if divide == 0:
                pass
            else:
                chunk_num = divide + 1
            print(var_name)
            print(sys.getsizeof(var))

        """
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

def fillna_mean_horse(df,mean = None,categorical = []):
    if mean is None:
        mean = df.mean(numeric_only = True)
    for col in df.columns:
        if col in categorical:
            mean.loc[col] = 0.0
    df = df.fillna(mean)
    return df

def fillna_mean_race(df,mean = None,categorical = []):
    means_stds = df.groupby("info_race_id").agg(["mean"]).reset_index()
    for col in df.columns:
        if col is "info_race_id":
            continue
        if col in categorical:
            mean[col] = 0
        ms = means_stds[col]
        df = df.join(ms,on = "info_race_id")
        df = df.fillna(df["mean"])
        df = df.drop("mean",axis=1)
    fillna_mean_horse(df,mean,categorical)
    return df

def check_duplicates(columns):
    print(set([x for x in columns if columns.count(x) > 1]))


if __name__ == "__main__":
    x_columns = ["addinfo_age_month","info_sex_code","info_sex_code"]
    check_duplicates(x_columns)
    y_columns = ["win_payoff"]
    odds_columns = ["linfo_win_odds","linfo_place_odds"]
    where = "rinfo_year > 14"

    #dp = DataProcessor()
    #dp.load_from_sql("db/output_v17.db",x_columns,y_columns,odds_columns,where = where)
    dp = load_from_database("db/output_v17.db",x_columns,y_columns,odds_columns,where = where)
    #dp = load_from_cache("test.pickle")
    #dp.dummy()
    dp.keep_separate_race_df()
    #dp.fillna_mean()
    #dp.normalize()
    #dp.to_race_panel()
    dp.summay()
    dp.save("test.pickle")
    train_x = dp.get(KEY_TRAIN_X)
    print(train_x)

