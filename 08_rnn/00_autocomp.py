#!/bin/env python

import tensorflow as tf
import numpy as np

char_arr = list('abcdefghijklmnopqrstuvwxyz')
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data_train = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']
seq_data_test = ['like', 'loop', 'dept', 'call', 'wolf']

def make_batch(seq_data):
	input_batch = []
	target_batch = []

	for seq in seq_data:
		input = [num_dic[n] for n in seq[:-1]]
		target = num_dic[seq[-1]]
		input_batch.append(np.eye(dic_len)[input])
		target_batch.append(target)

	return input_batch, target_batch

learning_rate = 0.01
n_hidden = 128
total_epoch = 30

n_step = 3
n_input = n_class = dic_len

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(
		tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=model, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	input_batch, target_batch = make_batch(seq_data_train)

	for epoch in range(total_epoch):
		_, loss = sess.run([optimizer, cost],
						feed_dict={X: input_batch, Y: target_batch})

		print('Epoch:', '%04d' % (epoch + 1),
			'cost =', '{:.6f}'.format(loss))

	print('Optimization Complete!')

	prediction = tf.cast(tf.argmax(model, 1), tf.int32)
	prediction_check = tf.equal(prediction, Y)
	accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

	seq_data = seq_data_test
	input_batch, target_batch = make_batch(seq_data)

	predict, accuracy_val = sess.run([prediction, accuracy],
									feed_dict={X: input_batch, Y: target_batch})

	predict_words = []
	for idx, val in enumerate(seq_data):
		last_char = char_arr[predict[idx]]
		predict_words.append(val[:3] + last_char)

	print('input:', [w[:3] + ' ' for w in seq_data])
	print('predict:', predict_words)
	print('accuracy:', accuracy_val)


