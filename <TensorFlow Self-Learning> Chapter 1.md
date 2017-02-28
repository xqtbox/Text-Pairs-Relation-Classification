# TensorFlow Chapter1 

[TOC]

## Tensor

TensorFlow 程序使用 tensor 数据结构来代表所有的数据。计算图中，操作间传递的数据都是 tensor。你可以把 TensorFlow tensor 看作是一个 n 维的数组或列表。「A tensor has a static type and dynamic dimensions」（一个 tensor 有一个静态的类型和一个动态的维数）。Tensor 可以再图中的节点 node 之间流通。

### Rank

在 TensorFlow 系统中，Tensor 的维数来被描述为 rank。但是 Tensor 的 rank 和矩阵的 rank 并不是同一个概念（矩阵的 rank 表示矩阵大小，比如 $n$ 阶矩阵就是 $n \times n$ 的矩阵，而 tensor 的 rank 其实是维数的意思）。Tensor 的 rank（sometimes referred to as *order* or *degree* or *n-dimension*）是 Tensor 维数数量的描述。比如，下面的 tensor（使用 Python 中 list 定义的）就是 2 阶：

```python
t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

当然，我们可以认为一个二阶的 tensor 就是我们平常所说的矩阵，一阶的 tensor 可以认为是一个向量。对于一个二阶的 tensor ，我们可以用语句 `t[i, j]` 来访问其中的任何元素。而对于三阶的 tensor ，我们可以用`t[i, j, k]` 来访问其中的任何元素。

| Rank | Math entity                      | Python example                           |
| ---- | -------------------------------- | ---------------------------------------- |
| 0    | Scalar (magnitude only)          | s = 483                                  |
| 1    | Vector (magnitude and direction) | v = [1.1, 2.2, 3.3]                      |
| 2    | Matrix (table of numbers)        | m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]    |
| 3    | 3-Tensor (cube of numbers)       | t = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]] |
| n    | n-Tensor (you get the idea)      | ....                                     |



### Shape

| Rank | Shape              | Dimension number | Example                                 |
| ---- | ------------------ | ---------------- | --------------------------------------- |
| 0    | []                 | 0-D              | A 0-D tensor. A scalar.                 |
| 1    | [D0]               | 1-D              | A 1-D tensor with shape [5].            |
| 2    | [D0, D1]           | 2-D              | A 2-D tensor with shape [3, 4].         |
| 3    | [D0, D1, D2]       | 3-D              | A 3-D tensor with shape [1, 4, 3].      |
| n    | [D0, D1, ... Dn-1] | n-D              | A tensor with shape [D0, D1, ... Dn-1]. |

Shape 可以通过 Python 中的整数列表或者元组（int list 或 tuples）来表示，也可以用 `TensorShape` 类。

 

### Data types

除了维度，Tensor 有一个数据类型属性，我们可以为一个 tensor 指定下列数据类型中的任意一个类型：

| Data type     | Python type   | Description                              |
| ------------- | ------------- | ---------------------------------------- |
| DT_FLOAT      | tf.float32    | 32 bits floating point.                  |
| DT_DOUBLE     | tf.float64    | 64 bits floating point.                  |
| DT_INT8       | tf.int8       | 8 bits signed integer.                   |
| DT_INT16      | tf.int16      | 16 bits signed integer.                  |
| DT_INT32      | tf.int32      | 32 bits signed integer.                  |
| DT_INT64      | tf.int64      | 64 bits signed integer.                  |
| DT_UINT8      | tf.uint8      | 8 bits unsigned integer.                 |
| DT_UINT16     | tf.uint16     | 16 bits unsigned integer.                |
| DT_STRING     | tf.string     | Variable length byte arrays. Each element of a Tensor is a byte array. |
| DT_BOOL       | tf.bool       | Boolean.                                 |
| DT_COMPLEX64  | tf.complex64  | Complex number made of two 32 bits floating points: real and imaginary parts. |
| DT_COMPLEX128 | tf.complex128 | Complex number made of two 64 bits floating points: real and imaginary parts. |
| DT_QINT8      | tf.qint8      | 8 bits signed integer used in quantized Ops. |
| DT_QINT32     | tf.qint32     | 32 bits signed integer used in quantized Ops. |
| DT_QUINT8     | tf.quint8     | 8 bits unsigned integer used in quantized Ops. |





## The Computational Graph

1. **Building the computational graph.**
2. **Running the computational graph.**

TensorFlow 程序通常被组织成一个构建阶段和一个执行阶段。在构建阶段，op 的执行步骤被描述成一个图。在执行阶段，使用会话执行执行图中的 op。

例如，通常在构建阶段创建一个图来表示和训练神经网络，然后在执行阶段反复执行图中的训练 op。



## Constant & Session 

```python
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
```

最后的输出内容为：

```python
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
```

注意到，输出的结果并不直接是 3.0 和 4.0，这是因为在这两个 node 并未被执行，在计算图中使用 **session** 来执行这两个 node：

```python
sess = tf.Session()
print(sess.run([node1, node2]))
```

此时我们会得到预期的输出：

```python
[3.0, 4.0]
```

我们还可以这么操作 Tensor：

```python
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))
```

最后的输出结果是：

```python
node3:  Tensor("Add_2:0", shape=(), dtype=float32)
sess.run(node3):  7.0
```

**Session** 对象在使用完之后需要关闭以释放资源。除了显式调用 **close()** 外，也可以使用 **「with」** 代码块来自动完成关闭动作。

```python
with tf.Session() as sess:
	result = sess.run([product])
	print(result)
```

在现实上，TensorFlow 将图形定义转换成分布式执行的操作，以充分利用可用的计算资源（如 CPU 或 GPU）。一般我们不需要显式指定使用 CPU 还是 GPU，TensorFlow 能自动检测。如果检测到 GPU，TensorFlow 会尽可能地利用找到的第一个 GPU 来执行操作。

如果机器上有超过一个可用的 GPU，除第一个外的其他 GPU 默认是不参与计算的。为了让 TensorFlow 使用这些 GPU，你必须将 op 明确指派给它们执行。`with...Device` 语句用来指派特定的 CPU 或 GPU 执行操作：

```python
with tf.Session() as sess:
	with tf.device("/gpu:1"):
		matrix1 = tf.constant([[3., 3.]])
		matrix2 = tf.constant([[2.], [2.]])
		product = tf.matmul(matrix1, matrix2)
		....
```

设备用字符串进行标识。目前支持的设备包括：

- `"/cpu:0"`：机器的 CPU。
- `"/gpu:0"`：机器的第一个 GPU，如果有的话。
- `"/gpu:1"`：机器的第二个 GPU，以此类推。

阅读使用 GPU 章节，了解 TensorFlow GPU 使用的更多信息。



## Placeholder

Placeholder ，顾名思义，占位符。有的时候，我们输入的数据并不是一个 constant 常量，而是需要我们用户输入的值，那么 placeholder 占位符的用法就非常地方便。

```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
add_and_triple = adder_node * 3.
with tf.Session() as sess:
	print(sess.run(adder_node, {a: 3, b:4.5}))
	print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
	print(sess.run(add_and_triple, {a: 3, b:4.5}))
```

输出结果为：

```python
7.5
[ 3.  7.]
22.5
```



## Variable

tf.constant 生成的是常量，我们是无法改变其值的；而 tf.Variable 生成的是变量，我们是可以通过 tf.assign() 等方法改变其值的。

```python
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	print(sess.run(linear_model, {x:[1,2,3,4]}))
	print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
	sess.run([fixW, fixb])
	print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
```

输出：

```python
[ 0.          0.30000001  0.60000002  0.90000004]
23.66
0.0
```



## tf.train API

**optimizers** 作用是缓慢地改变（改变的速度与其中的步长相关）变量来最小化损失函数。最简单的 **optimizer** 就是 **gradient descent**.

```python
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init) # reset values to incorrect defaults.
	for i in range(1000):
		sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

	print(sess.run([W, b]))
```
输出结果为：
```python
[array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]
```



## Complete program

```python
import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
	sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```

输出：

```python
W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
```



## tf.contrib.learn

`tf.contrib.learn` 是 TensorFlow 中用来简化机器学习的实现方法的一个库。其中包括：

- running training loops
- running evaluation loops
- managing data sets
- managing feeding

`tf.contrib.learn` 定义了大部分常用的模型。

**基本用法为：**

使用 `tf.contrib.learn` 来简化上一个程序：

```python
import tensorflow as tf
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns = features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size = 4, num_epochs = 1000)

# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
estimator.fit(input_fn = input_fn, steps = 1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
estimator.evaluate(input_fn = input_fn)
```

