#-*- coding:utf-8 -*-

import os
import urllib
import time
import zipfile
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup


def fetch_expanded_info(dir_path,username,password):
    create_directory(dir_path)
    target_url = "http://www.jrdb.com/member/datazip/Paci/index.html"
    ls = fetch_zip_list(target_url,username,password)
    print("Downloading expanded info to dir")
    for url in ls:
        print(url)
        fetch_zip(url,username,password,dir_path)
        time.sleep(0.1)

def fetch_zip_list(url,username,passoword):
    url_list = []
    r = requests.get(url,auth = HTTPBasicAuth(username,password))
    status = r.status_code
    print(status)
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
    print(r.status_code)
    content = r.content
    with open(file_path,"wb") as fp:
        fp.write(content)

def parse_filename(url):
    filename = url.split("/")[-1]
    return filename

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def unzip(dir_path,unzipped_path):
    ls = os.listdir(dir_path)
    ls = [os.path.join(dir_path,name) for name in ls]
    create_directory(unzipped_path)

    for zip_path in ls:
        with zipfile.ZipFile(zip_path,"r") as zf:
            print(zip_path)
            zf.extractall(path = unzipped_path)

if __name__=="__main__":
    username = input("Enter your username : ")
    password = input("Enter your password : ")

    download_path = "raw_text/zipped"
    unzipped_path = "raw_text/unzipped"
    fetch_expanded_info(download_path,username,password)
    unzip(download_path,unzipped_path)
