#-*- coding : UTF-8 -*-

import sqlite3

class Feature(object):
    def __init__(self,con):
        self.con = con

    def fetch_horse(self,target_columns,race_type = "win"):
        #copy target_columns
        target_columns = [v for v in target_columns]

        if race_type == "win":
            rt_col = "is_win"
        elif race_type == "place":
            rt_col = "is_place"
        else:
            rt_col = "is_win"
        target_columns.append(rt_col)
        columns_query = ",".join(target_columns)
        sql = "SELECT {0} FROM feature".format(columns_query)
        cur = self.con.execute(sql)
        for row in cur:
            yield row[:-1],row[-1]

    def fetch_xy(self,x_col,y_col,where = ""):
        #copy target_columns
        x_col = [v for v in x_col]
        y_col = [v for v in y_col]
        sep_idx = len(x_col)

        target_col = x_col + y_col
        columns_query = ",".join(target_col)
        if where != "":
            where_query = "WHERE {0}".format(where)
        else:
            where_query = ""
        sql = "SELECT {0} FROM feature{1}".format(columns_query,where_query)
        cur = self.con.execute(sql)
        for row in cur:
            yield row[:sep_idx],row[sep_idx:]

    def fetch_x(self,x_col,where = ""):
        #copy target_columns
        x_col = [v for v in x_col]

        target_col = x_col
        columns_query = ",".join(target_col)
        if where != "":
            where_query = "WHERE {0}".format(where)
        else:
            where_query = ""
        sql = "SELECT {0} FROM feature{1}".format(columns_query,where_query)
        cur = self.con.execute(sql)
        for row in cur:
            yield row

if __name__=="__main__":
    con = sqlite3.connect("db/output_v3.db")
    f = Feature(con)
    x_col = ["info_race_id","pre1_first_3f_delta"]
    y_col = ["is_win","is_place_win"]

    count = 0
    for x,y in f.fetch_with_xy(x_col,y_col):
        print(x)
        print(y)
        if count >= 100:
            break
        count += 1
