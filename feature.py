#-*- coding : UTF-8 -*-

import sqlite3

def create_feature(con):
    target_columns = []

    info_columns = ",".join(["hi.{0} as 'info_{0}'".format(c) for c in column_list(con,"horse_info")])
    target_columns.append(info_columns)

    result_columns = column_list(con,"result")
    for i in range(5):
        rn_query = ",".join(["r{1}.{0} as 'pre{1}_{0}'".format(c,i+1) for c in result_columns])
        target_columns.append(rn_query)

    columns_query = ",".join(target_columns)
    sql = """SELECT {0} FROM horse_info as hi
             LEFT JOIN result as r1 ON hi.pre1_result_id = r1.result_id
             LEFT JOIN result as r2 ON hi.pre2_result_id = r2.result_id
             LEFT JOIN result as r3 ON hi.pre3_result_id = r3.result_id
             LEFT JOIN result as r4 ON hi.pre4_result_id = r4.result_id
             LEFT JOIN result as r5 ON hi.pre5_result_id = r5.result_id;""".format(columns_query)

    print(sql)
    cur = con.execute(sql)
    
    for count,row in enumerate(cur):
        print(count)


def select_past_race(con,race_id):
    sql = "SELECT * FROM result WHERE race_id = '{0}'".format(race_id)
    cur = con.execute(sql)
    row = cur.fetchone()


def column_list(con,table_name):
    columns = []
    sql = "SELECT * FROM {0}".format(table_name)
    cur = con.execute(sql)
    for column in cur.description:
        name = column[0]
        columns.append(name)
    return columns
