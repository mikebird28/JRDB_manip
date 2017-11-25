#-*- coding:utf-8 -*-

from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dataset2

def show_similarity(target_idx,vectors):
    columns_name = []
    dic_1 = [u"芝",u"ダート",u"障害"]
    dic_2 = [u"短距離",u"短中距離",u"中距離",u"中長距離",u"長距離"]
    for i in range(165):
        dist = i//5
        typ = i%3
        course = i%11
        columns_name.append(u"{0}-{1:<3}-{2:<4}".format(course,dic_1[typ],dic_2[dist]))
    results = []
    for i in range(len(vectors)):
        cs = cos_similarity(vectors[target_idx],vectors[i])
        results.append((i,cs))
    results = sorted(results,key = lambda x: x[1])
    for t in results:
        print(u"{0} : {1}".format(columns_name[t[0]],t[1]))

def cos_similarity(x1,x2):
    cs = np.dot(x1,x2)/(np.linalg.norm(x1) * np.linalg.norm(x2))
    return cs

def top_n_k_keras(model,race_x,race_y,payoff,mode = "max"):
    counter = 0
    correct = 0
    rewards = 0
    for x,y,p in zip(race_x,race_y,payoff):
        pred = model.predict(x,verbose = 0)
        binary_pred = to_descrete(pred,mode = mode)

        y = np.array(y).ravel()
        p = np.array(p).ravel()
        c = np.dot(y,binary_pred)
        ret = np.dot(p,binary_pred)
        if c > 0:
            correct +=1
            rewards += ret
        counter += 1
    return (float(correct)/counter,float(rewards)/counter)


def top_n_k(model,race_x,race_y,payoff):
    counter = 0
    correct = 0
    rewards = 0
    for x,y,p in zip(race_x,race_y,payoff):
        if type(model) == KerasClassifier:
            pred = model.predict_proba(x,verbose = 0)[:,1].ravel()
        else:
            pred = model.predict_proba(x)[:,1].ravel()
        binary_pred = to_descrete(pred)

        y = np.array(y).ravel()
        p = np.array(p).ravel()
        c = np.dot(y,binary_pred)
        ret = np.dot(p,binary_pred)
        if c > 0:
            correct +=1
            rewards += ret
        counter += 1
    return (float(correct)/counter,float(rewards)/counter)

def top_n_k_regress(model,race_x,race_y,payoff):
    counter = 0
    correct = 0
    rewards = 0
    for x,y,p in zip(race_x,race_y,payoff):
        if type(model) == KerasRegressor:
            pred = model.predict(x,verbose = 0).ravel()
        else:
            pred = model.predict(x).ravel()
        binary_pred = to_descrete(pred)

        y = np.array(y).ravel()
        p = np.array(p).ravel()
        c = np.dot(y,binary_pred)
        ret = np.dot(p,binary_pred)
        if c > 0:
            correct +=1
            rewards += ret
        counter += 1
    return (float(correct)/counter,float(rewards)/counter)


def to_descrete(array,mode = "max"):
    res = np.zeros_like(array)
    if mode == "min":
        res[array.argmin(0)] = 1
    elif mode == "max":
        res[array.argmax(0)] = 1
    return res

def plot_importance(column,importance):
    x = range(len(column))
    plt.barh(x,importance)
    plt.yticks(x,column)
    plt.show()
    return

def show_importance(column,importance,sort = True,top_n = None):
    zipped = zip(column,importance)
    if sort:
        zipped = sorted(zipped,key = lambda x : x[1],reverse = True)

    counter = 0
    for f,i in zipped:
        if (top_n is not None) and (not counter < 20):
            break
        print("{0:<25} : {1:.5f}".format(f,i))
        counter += 1

if __name__ =="__main__":
    array = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(array)
    show_similarity(0,array)
    pass

