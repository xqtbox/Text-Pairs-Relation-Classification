import os
import randolph
import gensim
import logging
import time
import datetime
import tensorflow as tf
import numpy as np
from text_cnn import TextCNN
from tensorflow.contrib import learn
from gensim import corpora, models, similarities
from gensim.models import word2vec
from gensim.corpora import TextCorpus, MmCorpus, Dictionary
from tflearn.data_utils import to_categorical, pad_sequences

# Parameters
# ==================================================

FLAGS = tf.flags.FLAGS
BASE_DIR = randolph.cur_file_dir()

# Data loading params
tf.flags.DEFINE_string("text_dir", BASE_DIR + '/content.txt', "Data source for texts.")
tf.flags.DEFINE_string("vocabulary_data_dir", BASE_DIR + '/math.dict', "Data source for vocabulary dictionary.")
tf.flags.DEFINE_string("training_data_file", BASE_DIR + '/Model1_Training.txt', "Data source for the training data.")
tf.flags.DEFINE_string("test_data_file", BASE_DIR + '/Model1_Test.txt', "Data source for the test data.")

# Data parameters
tf.flags.DEFINE_string("MAX_SEQUENCE_LENGTH", 120, "每个文本的最长选取长度(padding的统一长度),较短的文本可以设短些.")
tf.flags.DEFINE_string("MAX_NB_WORDS", 5000, "整体词库字典中，词的多少，可以略微调大或调小.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

def max_seq_len_cal(content_indexlist):
	result = 0
	for item in content_indexlist:
		if len(item) > result:
			result = len(item)
	return result

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

def data_word2vec(inputFile, dictionary):
	def token_to_index(content, dictionary):
		list = []
		for item in content:
			if item != '<end>':
				list.append(dictionary.token2id[item])
		return list
	
	with open(inputFile) as fin:
		labels = []
		front_content_indexlist = []
		behind_content_indexlist = []
		for index, eachline in enumerate(fin):
			front_content = []
			behind_content = []
			line = eachline.strip().split('\t')
			label = line[2]
			content = line[3].strip().split(' ')
			
			end_tag = False
			for item in content:
				if item == '<end>':
					end_tag = True
				if end_tag == False:
					front_content.append(item)
				if end_tag == True:
					behind_content.append(item)
					
			labels.append(label)
			
			front_content_indexlist.append(token_to_index(front_content, dictionary))
			behind_content_indexlist.append(token_to_index(behind_content[1:], dictionary))
		total_line = index + 1
		
	class Data:
		def __init__(self, total_line, labels, front_content_indexlist, behind_content_indexlist):
			self.number = total_line
			self.labels = labels
			self.front_tokenindex = front_content_indexlist
			self.behind_tokenindex = behind_content_indexlist
			
	return Data(total_line, labels, front_content_indexlist, behind_content_indexlist)

def main():
	sentences = word2vec.LineSentence(FLAGS.text_dir)
	my_dict = Dictionary.load(FLAGS.vocabulary_data_dir)
	vocab_size = len(my_dict.items())

	print('---------------------------------------------------')
	print('Processing text dataset:')
	
	Training_Data = data_word2vec(inputFile=FLAGS.training_data_file, dictionary=my_dict) 
	Test_Data = data_word2vec(inputFile=FLAGS.test_data_file, dictionary=my_dict)
	
#	training_max_seq_len = max(max_seq_len_cal(Training_Data.front_tokenindex), max_seq_len_cal(Training_Data.behind_tokenindex))
#	test_max_seq_len = max(max_seq_len_cal(Test_Data.front_tokenindex), max_seq_len_cal(Training_Data.behind_tokenindex))
#	max_seq_len = max(training_max_seq_len, test_max_seq_len)
#	print('Max Sequence Length is: ' max_seq_len)

	print('---------------------------------------------------')
	print('Found %s training texts.' % Training_Data.number)
	data_training_front = pad_sequences(Training_Data.front_tokenindex, maxlen=FLAGS.MAX_SEQUENCE_LENGTH, value=0.)
	data_training_behind = pad_sequences(Training_Data.behind_tokenindex, maxlen=FLAGS.MAX_SEQUENCE_LENGTH, value=0.)
	labels_training = to_categorical(Training_Data.labels, nb_classes=2)	
	print('Shape of training data front tensor:', data_training_front.shape)
	print('Shape of training data behind tensor:', data_training_behind.shape)
	print('Shape of training label tensor:', labels_training.shape)

	print('---------------------------------------------------')
	print('Found %s test texts.' % Test_Data.number)
	data_test_front = pad_sequences(Test_Data.front_tokenindex, maxlen=FLAGS.MAX_SEQUENCE_LENGTH, value=0.)
	data_test_behind = pad_sequences(Test_Data.behind_tokenindex, maxlen=FLAGS.MAX_SEQUENCE_LENGTH, value=0.)
	labels_test = to_categorical(Test_Data.labels, nb_classes=2)
	print('Shape of test data front tensor:', data_test_front.shape)
	print('Shape of test data behind tensor:', data_test_behind.shape)
	print('Shape of test label tensor:', labels_test.shape)
	print('---------------------------------------------------')

	x_train_front = data_training_front
	x_train_behind = data_training_behind
	y_train = labels_training
	
	x_test_front = data_test_front
	x_test_behind = data_test_behind
	y_test = labels_test
	
	# Training
	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(
			allow_soft_placement = FLAGS.allow_soft_placement,
			log_device_placement = FLAGS.log_device_placement)
		sess = tf.Session(config = session_conf)
		with sess.as_default():
			cnn = TextCNN(
					sequence_length=FLAGS.MAX_SEQUENCE_LENGTH,
					num_classes=y_train.shape[1],
					vocab_size=vocab_size,
					embedding_size=FLAGS.embedding_dim,
					filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
					num_filters=FLAGS.num_filters,
					l2_reg_lambda=FLAGS.l2_reg_lambda)
			
			# Define Training procedure
			global_step = tf.Variable(0, name = "global_step", trainable = False)
			optimizer = tf.train.AdamOptimizer(1e-3)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

			# Keep track of gradient values and sparsity (optional)
			grad_summaries = []
			for g, v in grads_and_vars:
					if g is not None:
							grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
							sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
							grad_summaries.append(grad_hist_summary)
							grad_summaries.append(sparsity_summary)
			grad_summaries_merged = tf.summary.merge(grad_summaries)
			
			# Output directory for models and summaries
			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
			print("Writing to {}\n".format(out_dir))

			# Summaries for loss and accuracy
			loss_summary = tf.summary.scalar("loss", cnn.loss)
			acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
			
			# Train Summaries
			train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
			train_summary_dir = os.path.join(out_dir, "summaries", "train")
			train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

			# Dev summaries
			dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
			dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
			dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

			# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
					os.makedirs(checkpoint_dir)
			saver = tf.train.Saver(tf.global_variables(), max_to_keep = FLAGS.num_checkpoints)

			# Initialize all variables
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())

			def train_step(x_batch_front, x_batch_behind, y_batch):
					"""
					A single training step
					"""
					feed_dict = {
						cnn.input_x_front: x_batch_front,
						cnn.input_x_behind: x_batch_behind,
						cnn.input_y: y_batch,
						cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
					}
					_, step, summaries, loss, accuracy, scores, predictions, auc = sess.run(
							[train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy,
							 cnn.scores, cnn.predictions, cnn.AUC], feed_dict)
					time_str = datetime.datetime.now().isoformat()
					print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
					train_summary_writer.add_summary(summaries, step)

			def dev_step(x_batch_front, x_batch_behind, y_batch, writer=None):
					"""
					Evaluates model on a dev set
					"""
					feed_dict = {
						cnn.input_x_front: x_batch_front,
						cnn.input_x_behind:	x_batch_behind,
						cnn.input_y: y_batch,
						cnn.dropout_keep_prob: 1.0
					}
					step, summaries, loss, accuracy, scores, predictions, auc = sess.run(
							[global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.scores,
							 cnn.predictions, cnn.AUC], feed_dict)
					time_str = datetime.datetime.now().isoformat()
					print("{}: step {}, loss {:g}, acc {:g}, AUC {}".format(time_str, step, loss, accuracy, auc))
					if writer:
							writer.add_summary(summaries, step)

			# Generate batches
			batches = batch_iter(
					list(zip(x_train_front, x_train_behind, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
			# Training loop. For each batch...
			for batch in batches:
					x_batch_front, x_batch_behind, y_batch = zip(*batch)
					train_step(x_batch_front, x_batch_behind, y_batch)
					current_step = tf.train.global_step(sess, global_step)
					if current_step % FLAGS.evaluate_every == 0:
							print("\nEvaluation:")
							dev_step(x_test_front, x_test_behind, y_test, writer=dev_summary_writer)
							print("")
					if current_step % FLAGS.checkpoint_every == 0:
							path = saver.save(sess, checkpoint_prefix, global_step=current_step)
							print("Saved model checkpoint to {}\n".format(path))
	
main()

