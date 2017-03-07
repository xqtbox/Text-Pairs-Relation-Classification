# 基于 MNIST 数据集 的 「CNN」（tensorboard 绘图）
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import scipy

# Import itchat & threading
import itchat
import threading

# Create a running status flag
lock = threading.Lock()
running = False

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)
	
def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W, strides=1):
	return tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

def max_pool_2x2(x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
	
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

def nn_train(wechat_name, param):
	global lock, running
	# Lock
	with lock:
		running = True
		
	# 参数
	learning_rate, training_iters, batch_size, display_step = param
	
	# Import data
	mnist_data_path = 'MNIST_data/'
	mnist = input_data.read_data_sets(mnist_data_path, one_hot = True)
	
	# Network Parameters
	n_input = 28*28 # MNIST data input (img shape: 28*28)
	n_classes = 10 # MNIST total classes (0-9 digits)
	dropout = 0.75 # Dropout, probability to keep units
	
	with tf.name_scope('Input'):
		x = tf.placeholder(tf.float32, [None, n_input], name = 'input_x')
		y_ = tf.placeholder(tf.float32, [None, n_classes], name = 'target_y')
		keep_prob = tf.placeholder(tf.float32, name = 'keep_prob') #dropout (keep probability)

	def cnn_net(x, weights, biases, dropout):
		# Reshape input picture
		x_image = tf.reshape(x, [-1, 28, 28 ,1])
		
		# First Convolutional Layer
		conv_1 = add_layer(x_image, weights['conv1_w'], biases['conv1_b'], 'First_Convolutional_Layer', flag = 1)
		
		# First Pooling Layer
		pool_1 = max_pool_2x2(conv_1)
		
		# Second Convolutional Layer 
		conv_2 = add_layer(pool_1, weights['conv2_w'], biases['conv2_b'], 'Second_Convolutional_Layer', flag = 1)

		# Second Pooling Layer 
		pool_2 = max_pool_2x2(conv_2)

		# Densely Connected Layer
		pool_2_flat = tf.reshape(pool_2, [-1, weight_variable(weights['dc1_w']).get_shape().as_list()[0]])
		dc_1 = add_layer(pool_2_flat, weights['dc1_w'], biases['dc1_b'], 'Densely_Connected_Layer', flag = 0) 
		
		# Dropout
		dc_1_drop = tf.nn.dropout(dc_1, keep_prob)	
		
		# Readout Layer
		y = add_layer(dc_1_drop, weights['out_w'], biases['out_b'], 'Readout_Layer', flag = 0)
		
		return y
	
	# Store layers weight & bias
	weights = {
		# 5x5 conv, 1 input, 32 outputs
		'conv1_w': [5, 5, 1, 32],
		# 5x5 conv, 32 inputs, 64 outputs
		'conv2_w': [5, 5, 32, 64],
		# fully connected, 7*7*64 inputs, 1024 outputs
		'dc1_w': [7*7*64, 1024],
		# 1024 inputs, 10 outputs (class prediction)
		'out_w': [1024, n_classes]
	}

	biases = {
		'conv1_b': [32],
		'conv2_b': [64],
		'dc1_b': [1024],
		'out_b': [n_classes]
	}
	
	y = cnn_net(x, weights, biases, dropout)
	
	# Optimizer
	with tf.name_scope('cost'):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_,
						logits = y))
		tf.summary.scalar('cost', cost)
		tf.summary.histogram('cost', cost)
	
	# Train
	with tf.name_scope('train'):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	
	# Test
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

	
	# Train the model, and also write summaries.
	# Every 10th step, measure test-set accuracy, and write test summaries
	# All other steps, run train_step on training data, & add training summaries
	
	# Keep training until reach max iterations
	print('Wait for lock')
	with lock:
		run_state = running
	print('Start')
	
	step = 1
	while step * batch_size < training_iters and run_state:
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		# Run optimization op (backprop)
		sess.run(optimizer, feed_dict = {x: batch_x, y_: batch_y, keep_prob: dropout})
		if step % display_step == 0:	# Record execution stats
			run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			summary, _ = sess.run([merged, optimizer], feed_dict = 
									{x: batch_x, y_: batch_y, keep_prob: 1.}, 
									options = run_options, run_metadata = run_metadata)
			train_writer.add_run_metadata(run_metadata, 'step %d ' % step)
			train_writer.add_summary(summary, step)
			print('Adding run metadata for', step)

			summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict = 
											{x: batch_x, y_: batch_y, keep_prob: 1.})
			print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
				"{:.6f}".format(loss) + ", Training Accuracy= " + \
				"{:.5f}".format(acc))
			itchat.send("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
				"{:.6f}".format(loss) + ", Training Accuracy= " + \
						"{:.5f}".format(acc), wechat_name)
		else:
			summary, _ = sess.run([merged, optimizer], feed_dict = {x: batch_x, y_: batch_y, keep_prob: 1.})
			train_writer.add_summary(summary, step)
		step += 1
		with lock:
			run_state = running
	print("Optimization Finished!")
	itchat.send("Optimization Finished!", wechat_name)

	# Calculate accuracy for 256 mnist test images
	summary, acc = sess.run([merged, accuracy], feed_dict = 
							{x: mnist.test.images[:256], y_: mnist.test.labels[:256], 
							keep_prob: 1.} )
	text_writer.add_summary(summary)
	print("Testing Accuracy:", acc)
	itchat.send("Testing Accuracy: %s" % acc, wechat_name)

				
@itchat.msg_register([itchat.content.TEXT])
def chat_trigger(msg):
	global lock, running, learning_rate, training_iters, batch_size, display_step
	if msg['Text'] == u'开始':
		print('Starting')
		with lock:
			run_state = running
		if not run_state:
			try:
				threading.Thread(target=nn_train, args=(msg['FromUserName'], (learning_rate, training_iters, batch_size, display_step))).start()
			except:
				msg.reply('Running')
	elif msg['Text'] == u'停止':
		print('Stopping')
		with lock:
			running = False
	elif msg['Text'] == u'参数':
		itchat.send('lr=%f, ti=%d, bs=%d, ds=%d'%(learning_rate, training_iters, batch_size, display_step),msg['FromUserName'])
	else:
		try:
			param = msg['Text'].split()
			key, value = param
			print(key, value)
			if key == 'lr':
				learning_rate = float(value)
			elif key == 'ti':
				training_iters = int(value)
			elif key == 'bs':
				batch_size = int(value)
			elif key == 'ds':
				display_step = int(value)
		except:
			pass


if __name__ == '__main__':
	itchat.auto_login(hotReload=True)
	itchat.run()