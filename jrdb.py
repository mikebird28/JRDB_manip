
import argparse
import os
import sqlite3
import sys
import reader
import fetcher

ROOT_DIR = "./raw_text"
EXPANDED_INFO_PATH = "expanded_info"
TRAIN_INFO_PATH = "train_info"
RACE_INFO_PATH = "race_info"
LAST_INFO_PATH = "last_info"
PAYOFF_PATH = "payoff"
HORSE_INFO_PATH = "horse_info"
HORSE_RESULT_PATH = "horse_result"

def main():
    parser = argparse.ArgumentParser(description="script for generate JRDB database")
    sub_parser = parser.add_subparsers()

    fetch_parser = sub_parser.add_parser("fetch",help = "fetch zip files from JRDB server and extract ")
    fetch_parser.set_defaults(func = fetch)
    fetch_parser.add_argument("-d","--directory",default="raw_text")
    fetch_parser.add_argument("-u","--username",default=None)
    fetch_parser.add_argument("-p","--password",default=None)

    parse_parser = sub_parser.add_parser("parse", help = "parse csv files and insert to sql database")
    parse_parser.add_argument("output",nargs="?",default = "db/output.db")
    parse_parser.add_argument("-d","--directory",default="raw_text")
    parse_parser.add_argument("-c","--config",default="config/config.json")
    parse_parser.add_argument("-t","--is_test",action="store_true",default=False)
    parse_parser.set_defaults(func = parse)

    generate_parser = sub_parser.add_parser("generate", help = "generate features table from sql database")
    generate_parser.add_argument("inputs",nargs="?",default = "db/output.db")
    generate_parser.add_argument("-o","--output",default = "db/output.db")
    generate_parser.set_defaults(func = generate)

    args = parser.parse_args()
    args.func(args)

def fetch(args):
    username = args.username
    password = args.password
    if username == None:
        username = raw_input("> Enter your username\n")
    if password == None:
        password = raw_input("> Enter your password\n")
    is_valid = fetcher.is_valid_account(username,password)
    if not is_valid:
        raise Exception("unavalirable account")
    fetcher.fetch_all_datasets(ROOT_DIR,username,password)
    fetcher.decompress_all_datasets(ROOT_DIR)

def parse(args):
    db_con = sqlite3.connect(args.output)
    start = "031025"
    is_test = args.is_test

    #process about extra horse information
    ex_orm = reader.ExpandedInfoDatabase(db_con)
    csv_to_db(args,EXPANDED_INFO_PATH,"KKA",ex_orm,test_mode = is_test,start = start)

    #process about horse information
    hid_orm = reader.HorseInfoDatabase(db_con)
    csv_to_db(args,"horse_info","KYI",hid_orm,test_mode = is_test,start = start)

    ti_orm = reader.TrainingInfoDatabase(db_con)
    csv_to_db(args,TRAIN_INFO_PATH,"CYB",ti_orm,test_mode = is_test,start = start)

    #process about race information
    ri_orm = reader.RaceInfoDatabase(db_con)
    csv_to_db(args,RACE_INFO_PATH,"BAC",ri_orm,test_mode = is_test,start = start)

    #process about last inforamtion
    li_orm = reader.LastInfoDatabase(db_con)
    csv_to_db(args,LAST_INFO_PATH,"TYB",li_orm,test_mode = is_test,start = start)

    #process about payoff
    pd_orm = reader.PayoffDatabase(db_con)
    csv_to_db(args,"payoff", "HJC",pd_orm,test_mode = is_test,start = start)

    #process about past race result
    rd_orm = reader.ResultDatabase(db_con)
    csv_to_db(args,"horse_result", "SED",rd_orm,test_mode = is_test)

    db_con.close()
    #create feature table
    #reader.create_feature_table(db_con)

def generate(args):
    inp_con = sqlite3.connect(args.inputs)
    out_con = sqlite3.connect(args.output)
    reader.create_feature_table(inp_con,out_con)
    inp_con.close()
    out_con.close()
 
def csv_to_db(args,dir_name,file_prefix,orm,test_mode = False,start = None,end = None):
    path = os.path.join(args.directory,dir_name)
    if not os.path.exists(path):
        raise Exception("{0} directory doesn't exist".format(dir_name))
    files = [s.lower() for s in os.listdir(path)]
    files = filter(lambda s:s.startswith(file_prefix.lower()),files)

    if start is not None:
        if start[0:2] != "99":
            start = "1"+start
        start_int = int(start)
        def compare_date(s):
            s = s.replace(file_prefix,"")
            s = s.replace(".txt","")
            if s[0:2] != "99":
                s = "1"+s
            return int(s) >= start_int
        files = filter(compare_date,files)

    counter = 1
    for f in reversed(files):
        if test_mode and counter > 100:
            break
        sys.stdout.write("processing : {0}/{1}\r".format(counter,len(files)))
        sys.stdout.flush()
        file_path = os.path.join(path,f)
        with open(file_path,"r") as fp:
            orm.insert_file(fp)
        counter += 1
    print("")

if __name__=="__main__":
    main()
