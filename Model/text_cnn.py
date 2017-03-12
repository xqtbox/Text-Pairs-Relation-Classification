import randolph
import tensorflow as tf
import numpy as np
import multiprocessing
import gensim
from gensim.models import word2vec
from gensim.corpora import TextCorpus, MmCorpus, Dictionary
from sklearn.metrics import roc_auc_score

BASE_DIR = randolph.cur_file_dir()
TEXT_DATA_DIR = BASE_DIR + '/content.txt' 
WORD2VEC_DIR = BASE_DIR + '/math.model'
DICTIONARY_DIR = BASE_DIR + '/math.dict'

def word2vec_train(inputFile, outputFile, dictionary, vocab_size, embedding_size):
	sentences = word2vec.LineSentence(inputFile)
	my_dict = Dictionary.load(dictionary)
	
	# sg=0 -> CBOW model; sg=1 -> skip-gram model.
	model = gensim.models.Word2Vec(sentences, size=embedding_size, min_count=0,
																 sg=0, workers=multiprocessing.cpu_count())
	model.save(outputFile)
	
	Vector = np.zeros([vocab_size, embedding_size])
	for value, key in my_dict.items():
		Vector[value] = model[key]

	return Vector

class TextCNN(object):
	"""
	A CNN for text classification.
	Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
	"""
	def __init__(
		self, sequence_length, num_classes, vocab_size,
		embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

			# Placeholders for input, output and dropout
			self.input_x_front = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_front")
			self.input_x_behind = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_behind")
			self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
			self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

			# Keeping track of l2 regularization loss (optional)
			l2_loss = tf.constant(0.0)

			Vector = word2vec_train(TEXT_DATA_DIR, WORD2VEC_DIR, DICTIONARY_DIR, vocab_size, embedding_size)

			# Embedding layer
			with tf.device('/cpu:0'), tf.name_scope("embedding"):
				# 原采用的是随机生成正态分布的词向量。
				self.W = tf.Variable(
						tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
						name="W")
				# Vector 是通过自己的语料库训练而得到的词向量。
				# input_x_front 和 input_x_behind 共用 Vector。
#                    self.V = tf.Variable(Vector, name="W")
				self.embedded_chars_front = tf.nn.embedding_lookup(self.W, self.input_x_front)
				self.embedded_chars_behind = tf.nn.embedding_lookup(self.W, self.input_x_behind)
				
#                    self.embedded_chars_front = tf.nn.embedding_lookup(self.V, self.input_x_front)
#                    self.embedded_chars_behind = tf.nn.embedding_lookup(self.V, self.input_x_behind)
				
				self.embedded_chars_expanded_front = tf.expand_dims(self.embedded_chars_front, -1)
				self.embedded_chars_expanded_behind = tf.expand_dims(self.embedded_chars_behind, -1)
					
			# Create a convolution + maxpool layer for each filter size
			pooled_outputs_front = []
			pooled_outputs_behind = []

			for i, filter_size in enumerate(filter_sizes):
				with tf.name_scope("conv-maxpool-%s" % filter_size):
					# Convolution Layer
					filter_shape = [filter_size, embedding_size, 1, num_filters]
					W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
					b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
					conv_front = tf.nn.conv2d(
							self.embedded_chars_expanded_front,
							W,
							strides=[1, 1, 1, 1],
							padding="VALID",
							name="conv_front")
					
					conv_behind = tf.nn.conv2d(
							self.embedded_chars_expanded_behind,
							W,
							strides=[1, 1, 1, 1],
							padding="VALID",
							name="conv_behind")
							
					# Apply nonlinearity
					h_front = tf.nn.relu(tf.nn.bias_add(conv_front, b), name="relu_front")
					h_behind = tf.nn.relu(tf.nn.bias_add(conv_behind, b), name="relu_behind")
					# Maxpooling over the outputs
					pooled_front = tf.nn.max_pool(
							h_front,
							ksize=[1, sequence_length - filter_size + 1, 1, 1],
							strides=[1, 1, 1, 1],
							padding="VALID",
							name="pool_front")
					
					pooled_behind = tf.nn.max_pool(
							h_behind,
							ksize=[1, sequence_length - filter_size + 1, 1, 1],
							strides=[1, 1, 1, 1],
							padding="VALID",
							name="pool_behind")
					
					pooled_outputs_front.append(pooled_front)
					pooled_outputs_behind.append(pooled_behind)
							
			# Combine all the pooled features
			num_filters_total = num_filters * len(filter_sizes)
			self.h_pool_front = tf.concat(pooled_outputs_front, 3)
			self.h_pool_behind = tf.concat(pooled_outputs_behind, 3)
			self.h_pool_flat_front = tf.reshape(self.h_pool_front, [-1, num_filters_total])
			self.h_pool_flat_behind = tf.reshape(self.h_pool_behind, [-1, num_filters_total])

			self.h_pool_flat_combine = tf.concat([self.h_pool_flat_front, self.h_pool_flat_behind], 1)
			
			# Add dropout
			with tf.name_scope("dropout"):
				self.h_drop = tf.nn.dropout(self.h_pool_flat_combine, self.dropout_keep_prob)

			# Final (unnormalized) scores and predictions
			with tf.name_scope("output"):
				W = tf.get_variable(
						"W",
						shape=[num_filters_total*2, num_classes],
						initializer=tf.contrib.layers.xavier_initializer())
				b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
				l2_loss += tf.nn.l2_loss(W)
				l2_loss += tf.nn.l2_loss(b)
				self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
				self.predictions = tf.argmax(self.scores, 1, name="predictions")

			# CalculateMean cross-entropy loss
			with tf.name_scope("loss"):
				losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
				self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

			# Accuracy
			with tf.name_scope("accuracy"):
				correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
				self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

			# AUC
			with tf.name_scope("AUC"):
				 self.AUC = tf.contrib.metrics.streaming_auc(self.predictions, tf.argmax(self.input_y, 1))
