# -*- coding: utf-8 -*-

# Author : 'hxc'

# Time: 2021/1/29 3:19 PM

# File_name: 'training.py'

"""
Describe: this is a demo!
"""





import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.layers import *
from keras.models import *
import numpy as np
from keras.layers import Input, Dense, merge


class CNNLSTRMAttention(object):
    """cnn-lstm-attention"""

    def __init__(self,config):

        self.config = config
        self.data = pd.read_csv(self.config.data_file_path) #拿到数据

        #构造训练集与测试集
        self.build_dataset()
        self.build_network()
        self.build_train_model()

    def build_dataset(self):
        """构建训练集与测试集"""

        data_train = self.data.iloc[:int(self.data.shape[0] * self.config.train_data_rate), :]
        data_test = self.data.iloc[int(self.data.shape[0] * self.config.train_data_rate):, :]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data_train)
        data_train = scaler.transform(data_train)
        data_test = scaler.transform(data_test)

        self.X_train = np.array([data_train[i: i + self.config.seq_len, :] for i in range(data_train.shape[0] - self.config.seq_len)])
        self.y_train = np.array([data_train[i + self.config.seq_len, 0] for i in range(data_train.shape[0] - self.config.seq_len)])
        self.X_test = np.array([data_test[i: i + self.config.seq_len, :] for i in range(data_test.shape[0] - self.config.seq_len)])[-5:]

        self.y_test = np.array([data_test[i + self.config.seq_len, 0] for i in range(data_test.shape[0] - self.config.seq_len)])


    def build_network(self):
        """构建cnn-lstm网络"""

        self.inputs = Input(shape=(self.config.TIME_STEPS, self.config.INPUT_DIM))
        # drop1 = Dropout(0.3)(inputs)

        x = Conv1D(filters=64, kernel_size=1, activation='relu')(self.inputs)  # , padding = 'same'
        # x = Conv1D(filters=128, kernel_size=5, activation='relu')(output1)#embedded_sequences
        x = MaxPooling1D(pool_size=5)(x)
        x = Dropout(0.2)(x)
        lstm_out = Bidirectional(LSTM(self.config.lstm_units, activation='relu'), name='bilstm')(x)
        # lstm_out = LSTM(lstm_units,activation='relu')(x)

        # ATTENTION PART STARTS HERE
        attention_probs = Dense(128, activation='sigmoid', name='attention_vec')(lstm_out)
        # attention_mul=layers.merge([stm_out,attention_probs], output_shape],mode='concat',concat_axis=1))
        attention_mul = Multiply()([lstm_out, attention_probs])
        # attention_mul = merge([lstm_out, attention_probs],output_shape=32, name='attention_mul', mode='mul')

        self.output = Dense(1, activation='sigmoid')(attention_mul)
        # output = Dense(10, activation='sigmoid')(drop2)


    def build_train_model(self):
        """构建并训练模型"""

        self.model = Model(inputs=self.inputs, outputs=self.output)
        print(self.model.summary())
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(self.X_train, self.y_train, epochs=self.config.epochs, batch_size=self.config.batch_size, shuffle=False)
        self.model.save(self.config.save_model_path) #保存模型
        y_pred = self.model.predict(self.X_test)
        print(y_pred)
        print('MSE Train loss:', self.model.evaluate(self.X_train, self.y_train, batch_size=self.config.batch_size))
        print('MSE Test loss:', self.model.evaluate(self.X_test, self.y_test, batch_size=self.config.batch_size))
        plt.plot(self.y_test, label='test')
        plt.plot(y_pred, label='pred')
        plt.legend()
        plt.show()






if __name__ == "__main__":

    class Config():

        data_file_path = '/Users/mengqingyu/Desktop/quant/CNN/Binance_ETHUSDT_1m_1609459200000-1612310400000.csv'
        save_model_path = 'model—1m.h5'


        train_data_rate = 0.8
        output_dim = 1
        batch_size = 256
        epochs = 60
        seq_len = 5
        hidden_size = 128
        TIME_STEPS = 5
        INPUT_DIM = 5
        lstm_units = 64


    c= CNNLSTRMAttention(config=Config)














