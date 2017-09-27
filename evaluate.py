#-*- cofing:utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dataset

def top_n_k(model,features,race_x,race_y):
    counter = 0
    correct = 0
    for x,y in zip(race_x,race_y):
        x = pd.DataFrame(x,columns = features)
        x = dataset.fillna_zero(x)

        pred = model.predict_proba(x)[:,1]
        binary_pred = to_descrete(pred)

        y = np.array(y)
        c = np.dot(y,binary_pred)
        if c > 0:
            correct +=1
        counter += 1
    return float(correct)/counter


def to_descrete(array):
    res = np.zeros_like(array)
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


