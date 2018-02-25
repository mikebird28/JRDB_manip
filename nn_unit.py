#-*- coding:UTF-8 -*-

from keras.layers import Dense,Activation,Input,Dropout,Concatenate,Conv2D,Add,ZeroPadding2D,GaussianNoise,SpatialDropout1D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape,Flatten,Permute,Lambda,RepeatVector
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

def race_lstm(inputs_depth,lstm_units,activation = "relu",momentum = 0.0,dropout = 0.3):
    def __unit(x):
        lstm = Permute((1,3,2))(x)
        lstm = Reshape([18,inputs_depth])(lstm)
        lstm = LSTM(units = lstm_units,return_sequences = False)(lstm)
        lstm = RepeatVector(18)(lstm)
        lstm = Reshape([18,lstm_units,1])(lstm)
        x = Reshape([18,inputs_depth,1])(x)
        x = Concatenate(axis = 2)([x,lstm])

        x = Conv2D(inputs_depth,(1,inputs_depth + lstm_units),padding = "valid",kernel_initializer="he_normal")(x)
        x = Activation(activation)(x)
        x = BatchNormalization(axis = -1,momentum = momentum)(x)
        x = Dropout(dropout)(x)
        return x
    return __unit

def resnet(unit_size):
    def __unit():
        pass
    return __unit

def gate_unit(unit_size):
    def __unit():
        pass
