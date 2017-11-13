#!/bin/env python

import tensorflow as tf

hello = tf.constant('Hello, Tensorflow!')
print(hello)

a = tf.constant(10)
b = tf.constant(25)
c = tf.add(a, b)

print(a)
print(b)
print(c)

#sess = tf.Session()
#print(sess.run(hello))
#print(sess.run(c))
#sess.close()

with tf.Session() as sess:
	print(sess.run(hello))
	print(sess.run(c))
