#Ejemplo de red neuronal recurrente RNN- series temporales

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skelarn.reprocessing import MinMaxScaler

data = pd.read_csv('produccion-leche.csv', index_col='Month')
# crear indice
data.index = pd.to_datetime(data.index)
# data.plot()
# plt.show()
datos_entrens = data.head(150)
datos_pruebas = data.tail(18)
#normalizar los datos para usar tensorflow
normalizacion = MinMaxScaler()
entren_norm = normalizacion.fit_transform(datos_entrens)
pruebas_norma = normalizacion.transform(datos_pruebas)

#crear funcion lotes de datos
def lotes(datos_entrens, tam_lote, pasos):
    comienzo = np.random.randint(0,len(datos_entrens)-pasos)
    # indexar datos
    lote_y = np.array(datos_entrens[comienzo:comienzo+pasos+1]).reshape(1,pasos+1)
    return lote_y(:,:-1).reshape(-1,pasos,1), lote_y(:,1:).reshape(-1,pasos,1)
numero_entradas = 1
numero_pasos = 18
numero_neuronas = 120
numero_salidas = 1
tasa_aprend = 0.001
numero_entren = 5000
tam_lote = 1
x = tf.placeholder(tf.float32, [None, numero_pasos, numero_entradas])
y = tf.placeholder(tf.float32, [None, numero_pasos, numero_salidas])
capa = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTNCell(num_inits=numero_neuronas, activation=tf.nn.relu),output_size=numero_salidas)
salidas, estados =  tf.nn.dynamic_rnn(capa, x, dtype=tf.float32)
