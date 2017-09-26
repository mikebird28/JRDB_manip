#-*- coding:utf-8 -*-

import os
import sys
import time
import zipfile
import requests
import subprocess
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup


def fetch_expanded_info(dir_path,username,password):
    create_directory(dir_path)
    target_url = "http://www.jrdb.com/member/data/Jrdb/index.html"
    ls = fetch_list(target_url,username,password)
    print("Downloading expanded info to dir")
    length = len(ls)
    for count,url in enumerate(ls):
        sys.stdout.write("({0}/{1}) {2} ...".format(count+1,length,url))
        sys.stdout.flush()
        status = fetch_zip(url,username,password,dir_path)
        sys.stdout.write("{0}\n".format(status))
        sys.stdout.flush()
        time.sleep(1.0)

def fetch_list(url,username,passoword):
    url_list = []
    r = requests.get(url,auth = HTTPBasicAuth(username,password))
    status = r.status_code
    html = r.text
    soup = BeautifulSoup(html,"html.parser")
    links = soup.select("li a")

    url_path = "/".join(url.split("/")[:-1])
    for link in links:
        file_name = link["href"]
        url_list.append("/".join([url_path,file_name]))
    return url_list

def fetch_zip(url,username,password,directory):
    file_name = parse_filename(url)
    file_path = os.path.join(directory,file_name)
    r = requests.get(url,auth = HTTPBasicAuth(username,password))
    content = r.content
    with open(file_path,"wb") as fp:
        fp.write(content)
    return r.status_code

def parse_filename(url):
    filename = url.split("/")[-1]
    return filename

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract(dir_path,unzipped_path):
    ls = os.listdir(dir_path)
    ls = [os.path.join(dir_path,name) for name in ls]
    create_directory(unzipped_path)

    for path in ls:
        print(path)
        if path.endswith("zip"):
            unzip(path,unzipped_path)
        elif path.endswith("lzh"):
            unlzh(path,unzipped_path)
 
def unzip(file_path,unzipped_path):
    with zipfile.ZipFile(zip_path,"r") as zf:
        zf.extractall(path = unzipped_path)

def unlzh(file_path,unzipped_path):
    command = "lhasa xqw={0} {1}".format(unzipped_path,file_path)
    subprocess.call(command,shell=True)

if __name__=="__main__":
    username = raw_input("Enter your username : ")
    password = raw_input("Enter your password : ")

    download_path = "raw_text/zipped"
    unzipped_path = "raw_text/unzipped"
    fetch_expanded_info(download_path,username,password)
    extract(download_path,unzipped_path)
