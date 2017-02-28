# TensorFlow Chapter2 

[TOC]

## MNIST Complete program

```python
"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
```



## MNIST 数据集

每一个 MNIST 数据单元有两部分组成：一张包含手写数字的图片和一个对应的标签。我们把这些图片设为 “xs” ，把这些标签设为 “ys” 。训练数据集和测试数据集都包含 xs 和 ys，比如训练数据集的图片是 `mnist.train.images `，训练数据集的标签是 `mnist.train.labels`。

每一张图片包含 $28 \times 28$ 个像素点。我们可以用一个数字数组来表示这张图片：

<img src="https://farm3.staticflickr.com/2623/32883220381_738baaafc1_o.png" style="zoom:50%" />


我们把这个数组展开成一个向量，长度是 $28 \times 28 = 784$。如何展开这个数组（数字间的顺序）不重要，只要保持各个图片采用相同的方式展开。从这个角度来看，MNIST数据集的图片就是在 $784$ 维向量空间里面的点，并且拥有比较[复杂的结构](http://colah.github.io/posts/2014-10-Visualizing-MNIST/) (提醒: 此类数据的可视化是计算密集型的)。

展平图片的数字数组会丢失图片的二维结构信息。这显然是不理想的，最优秀的计算机视觉方法会挖掘并利用这些结构信息，我们会在后续教程中介绍。但是在这个教程中我们忽略这些结构，所介绍的简单数学模型，softmax 回归(softmax regression)，不会利用这些结构信息。


### mnist.train.xs

<img style="width:60%" src="https://farm6.staticflickr.com/5661/32883543751_24a5fb567c_o.png">

因此，在 MNIST 训练数据集中，`mnist.train.images` 是一个形状为$ [60000, 784]$ 的张量，第一个维度数字用来索引图片，第二个维度数字用来索引每张图片中的像素点。在此张量里的每一个元素，都表示某张图片里的某个像素的强度值，值介于 0 和 1 之间。

### mnist.train.ys

<img style="width:60%" src="https://farm1.staticflickr.com/588/32968241356_f4348c7299_o.png">

相对应的 MNIST 数据集的标签是介于 0 到 9 的数字，用来描述给定图片里表示的数字。为了用于这个教程，我们使标签数据是"one-hot vectors"。 一个 one-hot 向量除了某一位的数字是 1 以外其余各维度数字都是 0。所以在此教程中，数字 n 将表示成一个只有在第 n 维度（从 0 开始）数字为 1 的 10 维向量。比如，标签 0 将表示成([1,0,0,0,0,0,0,0,0,0,0])。因此， `mnist.train.labels` 是一个$[60000, 10]$ 的数字矩阵。



## Softmax Regression

对于 softmax Regression 模型可以用下面的图解释，对于输入的 xs 加权求和，再分别加上一个偏置量，最后再输入到 softmax 函数中：

<img style="width:55%" src="https://farm4.staticflickr.com/3895/32854820642_d19ae1626c_o.png">

如果把它写成一个等式，我们可以得到：

<img style="width:50%" src="https://farm1.staticflickr.com/535/32854819102_9c4ab2c67b_o.png">

我们也可以用向量表示这个计算过程：用矩阵乘法和向量相加。这有助于提高计算效率。（这同样也是一种更有效的思考方式）

<img style="width:50%" src="https://farm3.staticflickr.com/2713/32854821862_c02d2a372c_o.png">



## Cross-Entropy

为了训练我们的模型，我们首先需要定义一个指标来评估这个模型是好是坏。在 ML 中我们通常使用成本（cost）或损失（loss）函数来表示我们的模型离最好的理想模型差距有多大，然后尽可能地最小化这个指标。

一个常用的成本函数就是交叉熵（cross-entropy）。比较浅显地解释是，交叉熵用来衡量我们的模型预测值偏离实际真实值的程度大小。

最常见的例子就是在机器学习领域中的辨识问题（Classification）。假设说我们要训练机器辨识图片中的动物是狗还是猫，我们训练的模型可能会告诉我说这有 80% 机会是狗，20% 的机会是猫。假设正确的答案是狗，那么我们的模型有 80% 机率可以辨识出图片是狗到底够不够好? 如果是 85% 的话到底又好多少?

为了要让我们的模型最佳化，这个问题显得非常重要------因为我们需要一个方法来衡量我们模型的好坏。至于我们要最佳化什么，是需要根据这个模型最终是要被拿来做什么应用。这通常不会有一定的答案，结论是在某些时候，交叉熵就是我们所在乎的，而很多时候我们不知道训练出来的这个模型到底是如何被应用的时候，交叉熵就是一个很好的衡量标准。
$$
CrossEntropy = H_{y'}(y)= - \sum_i y'_i \cdot \log( y_i ) 
$$
简单地讲，交叉熵描述的是模型预测值 $y'_{i}$ 偏离实际真实值 $y_{i}$ 的程度大小。

当年香农 Shannon 创立信息论的时候，考虑的是每一次都是扔硬币，结果只有 2 个可能，所以用的是以 2 为底，发明了 bit 计量单位。而在 Tensorflow 软件里的实现，则是以 e 为底的 log。  

### tf.nn.softmax_cross_entropy_with_logits

Tensorflow 中有个经常用到的函数叫做 `tf.nn.softmax_cross_entropy_with_logits`。这个函数的实现并不在 Python 中，所以我用 Numpy 实现一个同样功能的函数进行比对，确认它使用的是以 e 为底的log。理由很简单，因为 Softmax 函数里使用了 e 的指数，所以当 Cross Entropy 也使用以 e 的 log，然后这两个函数放到一起实现，可以进行很好的性能优化。

```python
import tensorflow as tf
import numpy as np

# Make up some testing data, need to be rank 2

x = np.array([
		[0.,2.,1.],
		[0.,0.,2.]
		])
label = np.array([
		[0.,0.,1.],
		[0.,0.,1.]
		])

# Numpy part #

def softmax(logits):
    sf = np.exp(logits)
    sf = sf/np.sum(sf, axis=1).reshape(-1,1)
    return sf

def cross_entropy(softmax, labels):
	return -np.sum( labels * np.log(softmax), axis=1 )

def loss(cross_entropy):
	return np.mean( cross_entropy )

numpy_result = loss(cross_entropy( softmax(x), label ))

print(numpy_result)

# Tensorflow part #

g = tf.Graph()
with g.as_default():
	tf_x = tf.constant(x)
	tf_label = tf.constant(label)
	# labels 真实值 logits 预测值
	tf_ret = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = tf_x, labels = tf_label) )

with tf.Session(graph=g) as ss:
	tensorflow_result = ss.run([tf_ret])

print(tensorflow_result)
```

### tf.train.GradientDescentOptimizer

```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

我们使用 Tensorflow 中的 Gradient Descent algorithm 以 0.5 的学习速率最小化交叉熵。梯度下降算法是一个简单的学习过程，Tensorflow 只需要将每个变量一点点地往使成本不断降低的方向移动。当然 Tensorflow 也提供了[其他许多优化算法](https://www.tensorflow.org/api_guides/python/train#optimizers) ，我们所做的只需要简单地调整一行代码就可以使用其他的算法。

**TensorFlow在这里实际上所做的是，它会在后台给你描述的计算图里面增加一系列新的计算操作单元用于实现反向传播算法和梯度下降算法。然后，它返回给你的只是一个单一的操作，当运行这个操作时，它用梯度下降算法训练你的模型，微调你的变量，不断减少成本。**

```python
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

该循环的每个步骤中，我们都会随机抓取训练数据中的 100 个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行得到 `train_step` 的值。

使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是随机梯度下降训练。在理想情况下，我们希望用我们所有的数据来进行每一步的训练，因为这能给我们更好的训练结果，但显然这需要很大的计算开销。所以，每一次训练我们可以使用不同的数据子集，这样做既可以减少计算开销，又可以最大化地学习到数据集的总体特性。



## tf.InteractiveSession

这里，我们使用更加方便的InteractiveSession类。通过它，我们可以更加灵活地构建我们的代码。它能让我们在运行图的时候，插入一些计算图，这些计算图是由某些操作（operations）构成的。这对于工作在交互式环境中的人们来说非常便利，比如使用IPython。如果你没有使用InteractiveSession，那么你需要在启动 session 之前构建整个计算图，然后启动该计算图。
```python
import tensorflow as tf
sess = tf.InteractiveSession()
```



## Evaluating Our Model

### tf.argmax

​```python
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
```

`tf.argmax` 是一个非常有用的函数，它能给出某个 tensor 对象在某一维上的其数据最大值所在的索引值。由于标签向量是由 0 和 1组成，因此最大值 1 所在的索引位置就是类别标签，比如 `tf.argmax(y, 1)` 返回的是模型对任一输入 x 预测到的标签值，而 `tf.argmax(y_, 1)` 代表正确的标签，我们可以用 `tf.equal` 来检测我们的预测是否真实标签匹配（索引位置一样表示匹配）。其返回的是给我们一组布尔值，为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。例如，`[True, False, True, True]` 会变成`[1, 0, 1, 1]` ，取平均值后得到 `0.75`。

​```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
```

`tf.cast `  数据类型转换。

这个最终结果值应该大约是 91%。

这个结果好吗？嗯，并不太好。事实上，这个结果是很差的。这是因为我们仅仅使用了一个非常简单的模型。不过，做一些小小的改进，我们就可以得到97％的正确率。最好的模型甚至可以获得超过99.7％的准确率！（想了解更多信息，可以看看这个关于各种模型的[性能对比列表](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)。)