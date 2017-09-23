
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import sqlite3
import feature
import util


def main():
    x,y = generate_dataset()
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.1)
    forest = RandomForestClassifier()
    forest.fit(train_x,train_y)
    pred = forest.predict(test_x)
    print(classification_report(test_y,pred))

def generate_dataset():
    dataset_x = []
    dataset_y = []

    config = util.get_config("config.json")
    db_con = sqlite3.connect("output.db")

    f_orm = feature.Feature(db_con)
    target_columns = config.features
    for x,y in f_orm.fetch_horse(target_columns):
        dataset_x.append(x)
        dataset_y.append(y)
    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)
    return dataset_x,dataset_y

if __name__=="__main__":
    main()
