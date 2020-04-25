import tensorflow as tf
import numpy as np

#matrices
matrizA = np.random.uniform(0,20,(4,4))
matrizB = np.random.uniform(0,20,(4,1))
#placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# operaciones
suma = a + b
mult = a * b

print(matrizA)
print("\n")
print(matrizB)

with tf.Session() as sesion:
    sum = sesion.run(suma, feed_dict={a:matrizA,b:matrizB})
    multi = sesion.run(mult, feed_dict={a:matrizA,b:matrizB})
    print(sum)
    print("\n")
    print(multi)
