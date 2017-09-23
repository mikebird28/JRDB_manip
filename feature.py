#-*- coding : UTF-8 -*-

import sqlite3

class Feature(object):
    def __init__(self,con):
        self.con = con

    def fetch_horse(self,target_columns):
        target_columns.append("is_win")
        columns_query = ",".join(target_columns)
        sql = "SELECT {0} FROM feature".format(columns_query)
        cur = self.con.execute(sql)
        for row in cur:
            yield row[:-1],row[-1]

    def fetch_race(self,target_columns):
        fixed_target_columns = ["info_race_id"]
        fixed_target_columns.extend(target_columns)

        columns_query = ",".join(fixed_target_columns)
        sql = "SELECT {0} FROM feature ORDER BY info_race_id".format(columns_query)
        cur = self.con.execute(sql)

        is_fisrt = True
        bef_race_id = ""
        features = []

        for row in cur:
            race_id = row[0]
            if is_fisrt:
                is_fisrt = False
                bef_race_id = race_id
            features.append(row[1:])

            if bef_race_id != race_id:
                bef_race_id = race_id
                yield features
                features = []

def targets_for_test():
    pass
