#Ejemplo con MNIST - importar base de datos y mostrar una imagen

#importar librerias
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#leer datos
mnist = input_data.read_data_sets('data/', one_hot=True)
num_train = mnist.train.num_examples
num_test = mnist.test.num_examples

imagen = mnist.train.images[1]
imagen = imagen.reshape(28,28)
#mostrar imagen
plt.imshow(imagen)
plt.show()
