#「Linear Model」（tensorboard 绘图）
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(layoutname, inputs, in_size, out_size, act = None):
	with tf.name_scope(layoutname):
		with tf.name_scope('weights'):
			weights = tf.Variable(tf.random_normal([in_size, out_size]), name = 'weights')
			w_hist = tf.summary.histogram('weights', weights)
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name = 'biases')
			b_hist = tf.summary.histogram('biases', biases)
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)

		if act is None:
			outputs = Wx_plus_b
		else :
			outputs = act(Wx_plus_b)
		return outputs


x_data = np.linspace(-1, 1, 300)[:,np.newaxis]
noise = np.random.normal(0,0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('Input'):
	xs = tf.placeholder(tf.float32, [None, 1], name = "input_x")
	ys = tf.placeholder(tf.float32, [None, 1], name = "target_y")


l1 = add_layer("first_layer", xs, 1, 10, act = tf.nn.relu)
l1_hist = tf.summary.histogram('l1', l1)

y = add_layer("second_layout", l1, 10, 1, act = None)
y_hist = tf.summary.histogram('y', y)

with tf.name_scope('loss'): 
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - y), 
							reduction_indices = [1]))
	tf.summary.histogram('loss ', loss)
	tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

with tf.Session() as sess:
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(x_data, y_data)
	plt.ion()
	plt.show()
	
	writer = tf.summary.FileWriter('logs/', sess.graph)
	sess.run(init)
	
	for train in range(1000):
		sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})
		if train % 50 == 0:
			try:
				ax.lines.remove(lines[0])
			except Exception:
				pass
			result = sess.run(merged, feed_dict = {xs: x_data, ys: y_data})
			summary_str = result
			writer.add_summary(summary_str, train)
			print(train, sess.run(loss, feed_dict = {xs: x_data, ys: y_data}))
			
			prediction_value = sess.run(y, feed_dict = {xs: x_data})
			lines = ax.plot(x_data, prediction_value, 'r-', lw = 5)
			plt.pause(1)