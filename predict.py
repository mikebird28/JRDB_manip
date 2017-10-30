#-*- coding:UTF-8 -*-

import datetime
import requests
import zipfile
import io
import sqlite3
from requests.auth import HTTPBasicAuth
import reader
import dataset2
import util
import dqn

def main():
    config = util.get_config("config/config.json")

    date = datetime.date.today()
    date = date.replace(day = 29)
    username = raw_input(">> Enter your username: ")
    password = raw_input(">> Enter your password: ")
    csv = fetch_csv(date,username,password)
    con = create_database(csv)
    dqn.predict(con,config)

def fetch_csv(date,username,password):
    pinfo_url = "http://www.jrdb.com/member/datazip/Paci/{0}/PACI{1}.zip".format(date.year,datecode(date))
    linfo_url = "http://www.jrdb.com/member/datazip/Tyb/{0}/TYB{1}.zip".format(date.year,datecode(date))
    file_dict = {}

    linfo_r = requests.get(linfo_url,auth = HTTPBasicAuth(username,password))
    status = linfo_r.status_code
    if status != 200:
        print(status)
        raise Exception("Cannot fetch zipfile")
    linfo_bytes = io.BytesIO(linfo_r.content)
    linfo_zip  = zipfile.ZipFile(linfo_bytes)
    for filename in linfo_zip.namelist():
        fp = linfo_zip.open(filename)
        content = fp.read()
        file_dict[filename] = io.BytesIO(content)
        fp.close()
    linfo_zip.close()

    pinfo_r = requests.get(pinfo_url,auth = HTTPBasicAuth(username,password))
    status = pinfo_r.status_code
    if status != 200:
        raise Exception("Cannot fetch zipfile")
    pinfo_bytes = io.BytesIO(pinfo_r.content)
    pinfo_zip  = zipfile.ZipFile(pinfo_bytes)
    for filename in pinfo_zip.namelist():
        fp = pinfo_zip.open(filename)
        content = fp.read()
        file_dict[filename] = io.BytesIO(content)
        fp.close()
    pinfo_zip.close()
    return file_dict

def create_database(csv_dict):
    db_con = sqlite3.connect(":memory:")
    orm_dict = {
        "CYB" : reader.TrainingInfoDatabase(db_con),
        "BAC" : reader.RaceInfoDatabase(db_con),
        "TYB" : reader.LastInfoDatabase(db_con),
        "HJC" : reader.PayoffDatabase(db_con),
        "KYI" : reader.HorseInfoDatabase(db_con),
        "ZED" : reader.ResultDatabase(db_con),
        "KKA" : reader.ExpandedInfoDatabase(db_con),
    }

    for k,v in csv_dict.items():
        try:
            prefix = k[0:3]
            orm = orm_dict[prefix]
            orm.insert_file(v)
        except KeyError:
            pass
    reader.create_predict_table(db_con)
    return db_con

def datecode(date):
    year = str(date.year)
    month = str(date.month)
    day = str(date.day)
    datecode = year[2:4]+month+day
    return datecode


if __name__ == "__main__":
    main()
