# TensorFlow Tensorboard

本文主要介紹 TensorFlow 的 Tensorboard 模塊。

Tensorboard 可以看做是我們構建的Graph 的可視化工具，對於我們初學者理解網絡架構、每層網絡的細節都是很有幫助的。由於前幾天剛接觸 TensorFlow，所以在嘗試學習 Tensorboard 的過程中，遇到了一些問題。在此基礎上，參考了 TensorFlow 官方的 Tensorboard Tutorials 以及網上的一些文章。由於前不久 TensorFlow 1.0 剛發佈，網上的一些學習資源或者是 tensorboard 代碼在新的版本中並不適用，所以自己改寫并實現了官方網站上提及的三個實例的 Tensorboard 版本 ：
1. 最基礎簡單的「linear model」
2. 基於 MNIST 手寫體數據集的 「softmax regression」模型
3. 基於 MNIST 手寫體數據集的「CNN」模型

文章不會詳細介紹 TensorFlow 以及 Tensorboard 的知識，主要是模型的代碼以及部分模型實驗截圖。

注意：文章前提默認讀者們知曉 TensorFlow，知曉 Tensorboard，以及 TensorFlow 的一些主要概念「Variables」、「placeholder」。還有，默認你已經將需要用到的 MNIST 數據集下載到了你代碼當前所在文件夾。

## Environment

**OS: macOS Sierra 10.12.x**

**Python Version: 3.4.x**

**TensorFlow: 1.0**


## Tensorboard

Tensorboard有幾大模塊：

- SCALARS：记录單一變量的，使用 `tf.summary.scalar()` 收集構建。
- IMAGES：收集的图片数据，当我们使用的数据为图片时（选用）。
- AUDIO：收集的音频数据，当我们使用数据为音频时（选用）。
- GRAPHS：构件图，效果图类似流程图一样，我们可以看到数据的流向，使用`tf.name_scope()`收集構建。
- DISTRIBUTIONS：用于查看变量的分布值，比如 W（Weights）变化的过程中，主要是在 0.5 附近徘徊。
- HISTOGRAMS：用于记录变量的历史值（比如 weights 值，平均值等），并使用折线图的方式展现，使用`tf.summary.histogram()`进行收集構建。

## Examples

- 最簡單的線性回歸模型（tensorboard 繪圖）

```python
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
			summary_str = sess.run(merged, feed_dict = {xs: x_data, ys: y_data})
			writer.add_summary(summary_str, train)

			print(train, sess.run(loss, feed_dict = {xs: x_data, ys: y_data}))
			
			prediction_value = sess.run(y, feed_dict = {xs: x_data})
			lines = ax.plot(x_data, prediction_value, 'r-', lw = 5)
			plt.pause(1)
```

- 基於 Softmax Regressions 的 MNIST 數據集（tensorboard 繪圖）

```python
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
```

- 基於 CNN 的 MNIST 數據集（tensorboard 繪圖）

```python
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
```
可能對於最後一個模型 CNN 的代碼，需要一些 CNN 卷積神經網絡的一些知識。例如什麼是卷積、池化，還需要了解 TensorFlow 中用到的相應函數，例如`tf.nn.conv2d()`，`tf.nn.max_pool()`，這裡不再贅述。

貼上最後一個模型的部分截圖：

- 代碼部分：

![](https://farm4.staticflickr.com/3813/33035149741_c90aa2c7a7_o.png)

說明：右邊是 CNN 網絡訓練的步數以及對應的結果，細心的同學可能發現了，這個程序跑了我接近十六個小時（不知道正確不正確）😂。但是昨天晚上我應該是，十點開始跑這個程序，掛在實驗室，早上十點過來看完成了。總之，你們可以修改那個 range(20000)，請量力而為。

---

上述代碼運行完成之後，命令行中跳轉到代碼生成的「train」文件夾中（其和代碼文件存在于同一文件夾中），然後輸入 `tensorboard --logdir .`，等待程序反應之後，瀏覽器訪問`localhost:6006`（當然你也可以自己定義端口）。如果不出意外，你會得到以下內容：

- Scalars:

  ![](https://farm3.staticflickr.com/2524/33035142071_0bfc4e428c_o.png)

- Graphs:

  ![](https://farm1.staticflickr.com/668/33035146431_d86b30092d_o.png)

- Distributions:

  ![](https://farm4.staticflickr.com/3938/33035148401_377afc152d_o.png)

- Histograms:

  ![](https://farm3.staticflickr.com/2943/33035143981_cfa43b9962_o.png)

關於各個模塊的作用，以及各個變量的意義，我在此就不再贅述了。