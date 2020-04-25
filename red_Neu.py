# Red neuronal

import tensorflow as tf
import numpy as np
caracteristicas = 10
neuronas = 4
x = tf.placeholder(tf.float32,(None, caracteristicas))
w = tf.Variable(tf.random.normal([caracteristicas, neuronas]))
b = tf.Variable(tf.ones([neuronas]))
multiplicacion = tf.matmul(x,w)
z = tf.add(multiplicacion,b)

#obtener resultado final de la neurona
activacion = tf.sigmoid(z)
# inicializar variables
inicializar = tf.global_variables_initializer()
valores_x = np.random.random([1,caracteristicas])
# sesiones
with tf.Session() as sesion:
    sesion.run(inicializar)
    resultado = sesion.run(activacion, feed_dict = {x:valores_x})
    print(resultado)
