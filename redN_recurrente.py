# RNN con una capa de 3 neuronas desenrollada 2 veces
import numpy as np
import tensorflow as tf
numero_entradas = 2
neuronas = 3
#valores de entrada
x0 = tf.placeholder(tf.float32, [None,numero_entradas])
x1 = tf.placeholder(tf.float32, [None,numero_entradas])
#pesos
wx = tf.Variable(tf.random_normal(shape=[numero_entradas, neuronas]))
wy = tf.Variable(tf.random_normal(shape=[neuronas, neuronas]))
#bias
b = tf.Variable(tf.zeros([1,neuronas]))
#funciones de salida
y0 = tf.tanh(tf.matmul(x0,wx) + b)
y1 = tf.tanh(tf.matmul(y0,wy) + tf.matmul(x1,wx) + b)
lote_x0 = np.array([[0,1],[2,3],[4,5]])
lote_x1 = np.array([[2,4],[3,9],[4.1]])
#variable de inicializacion
init = tf.global_variables_initializer()

#sesion

with tf.Session() as sesion:
    sesion.run(init)
    salida_y0, salida_y1 = sesion.run([y0,y1], feed_dict={x0:lote_x0, x1:lote_x1})
print(salida_y0)
print(salida_y1)
#print(salida_y1)
