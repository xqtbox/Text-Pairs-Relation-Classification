# 基于 MNIST 数据集 的 「CNN」（tensorboard 绘图）
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)
	
def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)

def add_layer(input_tensor, weights_shape, biases_shape, layer_name, act = tf.nn.relu, flag = 1):
	"""Reusable code for making a simple neural net layer.

	It does a matrix multiply, bias add, and then uses relu to nonlinearize.
	It also sets up name scoping so that the resultant graph is easy to read,
	and adds a number of summary ops.
	"""
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			weights = weight_variable(weights_shape)
			variable_summaries(weights)
		with tf.name_scope('biases'):
			biases = bias_variable(biases_shape)
			variable_summaries(biases)
		with tf.name_scope('Wx_plus_b'):
			if flag == 1:
				preactivate = tf.add(conv2d(input_tensor, weights), biases)
			else:
				preactivate = tf.add(tf.matmul(input_tensor, weights), biases)
			tf.summary.histogram('pre_activations', preactivate)
		if act == None:
			outputs = preactivate
		else:
			outputs = act(preactivate, name = 'activation')
			tf.summary.histogram('activation', outputs)
		return outputs

def main():
	# Import data
	mnist_data_path = 'MNIST_data/'
	mnist = input_data.read_data_sets(mnist_data_path, one_hot = True)
	
	with tf.name_scope('Input'):
		x = tf.placeholder(tf.float32, [None, 28*28], name = 'input_x')
		y_ = tf.placeholder(tf.float32, [None, 10], name = 'target_y')

	# First Convolutional Layer
	x_image = tf.reshape(x, [-1, 28, 28 ,1])
	conv_1 = add_layer(x_image, [5, 5, 1, 32], [32], 'First_Convolutional_Layer', flag = 1)
	
	# First Pooling Layer
	pool_1 = max_pool_2x2(conv_1)
	
	# Second Convolutional Layer 
	conv_2 = add_layer(pool_1, [5, 5, 32, 64], [64], 'Second_Convolutional_Layer', flag = 1)

	# Second Pooling Layer 
	pool_2 = max_pool_2x2(conv_2)

	# Densely Connected Layer
	pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64])
	dc_1 = add_layer(pool_2_flat, [7*7*64, 1024], [1024], 'Densely_Connected_Layer', flag = 0) 
	
	# Dropout
	keep_prob = tf.placeholder(tf.float32)
	dc_1_drop = tf.nn.dropout(dc_1, keep_prob)
	
	# Readout Layer
	y = add_layer(dc_1_drop, [1024, 10], [10], 'Readout_Layer', flag = 0)
	
	# Optimizer
	with tf.name_scope('cross_entroy'):
		cross_entroy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_,
						logits = y))
		tf.summary.scalar('cross_entropy', cross_entroy)
		tf.summary.histogram('cross_entropy', cross_entroy)
	
	# Train
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entroy)
	
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)
		
	sess = tf.InteractiveSession()
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter('train/', sess.graph)
	test_writer = tf.summary.FileWriter('test/')
	tf.global_variables_initializer().run()

	def feed_dict(train):
		if train:
			batch_xs, batch_ys = mnist.train.next_batch(100)
			k = 0.5
		else:
			batch_xs, batch_ys = mnist.test.images, mnist.test.labels
			k = 1.0
		return {x: batch_xs, y_: batch_ys, keep_prob: k}
		
	for i in range(20000):
		if i % 100 == 0:
			# Test
			summary, acc = sess.run([merged, accuracy], feed_dict = feed_dict(False))
			test_writer.add_summary(summary, i)
			print("step %d, training accuracy %g" %(i, acc))
		else:
			# Train
			summary, _ = sess.run([merged, train_step], feed_dict = feed_dict(True))
			train_writer.add_summary(summary, i)			

main()