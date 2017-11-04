#!/bin/env python
#coding: utf-8

import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 3])
print(X)

x_data = [[1, 2, 3], [4, 5, 6]]

W = tf.Variable(tf.random_normal([3, 2]))

# 책 본문 오류
# 책에는 b=[2, 1]로 있는데, 이러면 x_data의 행 개수를 2로 강요하게 되어서
# 행을 3개 이상 넣으면 오류가 발생함
# [2] 혹은 [1, 2]로 하는게 맞는듯함
#b = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1, 2]))

expr = tf.matmul(X, W) + b
print(expr)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	print(x_data)
	print(sess.run(W))
	print(sess.run(b))

	print(sess.run(expr, feed_dict={X: x_data}))
