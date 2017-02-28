# 基于 MNIST 数据集 的 「softmax regress」（tensorboard 绘图） 
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def add_layer(layoutname, inputs, in_size, out_size, act = None):
	with tf.name_scope(layoutname):
		with tf.name_scope('weights'):
			weights = tf.Variable(tf.zeros([in_size, out_size]), name = 'weights')
			w_hist = tf.summary.histogram("weights", weights)
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros(out_size), name = 'biases')
			b_hist = tf.summary.histogram("biases", biases)
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)
		
		if act is None:
			outputs = Wx_plus_b
		else:
			outputs = act(Wx_plus_b)
		return outputs
		
# Import data
mnist_data_path = 'MNIST_data/'
mnist = input_data.read_data_sets(mnist_data_path, one_hot = True)

with tf.name_scope('Input'):
	x = tf.placeholder(tf.float32, [None, 28 * 28], name = 'input_x')
	y_ = tf.placeholder(tf.float32, [None, 10], name = 'target_y')

y = add_layer("hidden_layout", x, 28*28, 10, act = tf.nn.softmax)
y_hist = tf.summary.histogram('y', y)

# labels 真实值 logits 预测值
with tf.name_scope('loss'):
	cross_entroy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_,
					logits = y))
	tf.summary.histogram('cross entropy', cross_entroy)
	tf.summary.scalar('cross entropy', cross_entroy)

with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entroy)

# Test trained model
with tf.name_scope('test'):
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

with tf.Session() as sess:
	#logpath = r'/Users/randolph/PycharmProjects/TensorFlow/logs'
	writer = tf.summary.FileWriter('logs/', sess.graph)
	sess.run(init)

	for i in range(1000):
		if i % 10 == 0:
			feed = {x: mnist.test.images, y_: mnist.test.labels}
			result = sess.run([merged, accuracy], feed_dict = feed)
			summary_str = result[0]
			acc = result[1]
			writer.add_summary(summary_str, i)
			print(i, acc)
		else:
			batch_xs, batch_ys = mnist.train.next_batch(100)
			feed = {x: batch_xs, y_: batch_ys}
			sess.run(train_step, feed_dict = feed)

	print('final result: ', sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))