import tensorflow as tf
import numpy as np
array = np.arange(0,30,4)
g = tf.constant(array)
mensaje1 = tf.constant("hola")
mensaje2 = tf.constant("mundo")
resultado = (mensaje1 + mensaje2)
# with tf.Session() as sesion:
#     resultado = sesion.run(mensaje1 + mensaje2)
a = tf.constant(10)
b = tf.constant(4)
c = (a+b)
operaciones = [g, resultado, c ]
with tf.Session() as sesion:
    for op in operaciones:
        print(sesion.run(op))
        print("\n")
