import os
import randolph
import time
import datetime
import tensorflow as tf
from text_cnn import TextCNN
import data_helpers
import model_exports

# Parameters
# ==================================================

FLAGS = tf.flags.FLAGS
BASE_DIR = randolph.cur_file_dir()

Subset = '1' # 需要训练和测试的子集
TRAININGSET_DIR = BASE_DIR + '/Model Training' + '/Model' + Subset + '_Training.txt'
VALIDATIONSET_DIR = BASE_DIR + '/Model Validation' + '/Model' + Subset + '_Validation.txt'
TESTSET_DIR = BASE_DIR + '/Model Test' + '/Model' + Subset + '_Test.txt'

# Data loading params
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data.")
tf.flags.DEFINE_string("test_data_file", TESTSET_DIR, "Data source for the test data.")

# Data parameters
tf.flags.DEFINE_string("MAX_SEQUENCE_LENGTH", 450, "每个文本的最长选取长度(padding的统一长度),较短的文本可以设短些.")
tf.flags.DEFINE_string("MAX_NB_WORDS", 10000, "整体词库字典中，词的多少，可以略微调大或调小.")

# Model Hyperparameterss
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
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
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device Model1/train_cnn.py:39soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

# Model export parameters
tf.flags.DEFINE_string("input_graph_name", "input_graph.pb", "Graph input file of the graph to export")
tf.flags.DEFINE_string("output_graph_name", "output_graph.pb", "Graph output file of the graph to export")
tf.flags.DEFINE_string("output_node", "output/predictions", "The output node of the graph")

def main():
	# Data Preparation
	# ==================================================

	# Load data
	print('Loading data...')
	
	print('Training data processing...')
	x_train_front, x_train_behind, y_train = \
		data_helpers.load_data_and_labels(FLAGS.training_data_file, FLAGS.MAX_SEQUENCE_LENGTH, FLAGS.embedding_dim)

	print('Validation data processing...')
	x_validation_front, x_validation_behind, y_validation = \
		data_helpers.load_data_and_labels(FLAGS.validation_data_file, FLAGS.MAX_SEQUENCE_LENGTH, FLAGS.embedding_dim)

	# print('Test data processing...')
	# x_test_front, x_test_behind, y_test = \
	# 	data_helpers.load_data_and_labels(FLAGS.test_data_file, FLAGS.MAX_SEQUENCE_LENGTH, FLAGS.embedding_dim)

	vocab_size = data_helpers.load_vocab_size()
	pretrained_word2vec_matrix = data_helpers.load_word2vec_matrix(vocab_size, FLAGS.embedding_dim)

	# Training
	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(
			allow_soft_placement=FLAGS.allow_soft_placement,
			log_device_placement=FLAGS.log_device_placement)
		# session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn = TextCNN(
					sequence_length=FLAGS.MAX_SEQUENCE_LENGTH,
					num_classes=y_train.shape[1],
					vocab_size=vocab_size,
					embedding_size=FLAGS.embedding_dim,
					filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
					num_filters=FLAGS.num_filters,
					l2_reg_lambda=FLAGS.l2_reg_lambda,
					pretrained_embedding=pretrained_word2vec_matrix)

			# Define Training procedure
			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-3)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name="train_op")

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

			# Validation summaries
			validation_summary_op = tf.summary.merge([loss_summary, acc_summary])
			validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
			validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

			# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
					os.makedirs(checkpoint_dir)
			saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

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
					_, step, summaries, loss, accuracy = sess.run(
							[train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
					time_str = datetime.datetime.now().isoformat()
					print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
					train_summary_writer.add_summary(summaries, step)

			def validation_step(x_batch_front, x_batch_behind, y_batch, writer=None):
					"""
					Evaluates model on a validation set
					"""
					feed_dict = {
						cnn.input_x_front: x_batch_front,
						cnn.input_x_behind:	x_batch_behind,
						cnn.input_y: y_batch,
						cnn.dropout_keep_prob: 1.0
					}
					step, summaries, scores, predictions, topKPreds, loss, accuracy, auc = sess.run(
							[global_step, validation_summary_op, cnn.scores, cnn.predictions,
							 cnn.topKPreds, cnn.loss, cnn.accuracy, cnn.AUC], feed_dict)
					time_str = datetime.datetime.now().isoformat()
					print("{}: step {}, loss {:g}, acc {:g}, AUC {}"
						  .format(time_str, step, loss, accuracy, auc))
					if writer:
						writer.add_summary(summaries, step)

			# Generate batches
			batches = data_helpers.batch_iter(
					list(zip(x_train_front, x_train_behind, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
			# Training loop. For each batch...
			for batch in batches:
					x_batch_front, x_batch_behind, y_batch = zip(*batch)
					train_step(x_batch_front, x_batch_behind, y_batch)
					current_step = tf.train.global_step(sess, global_step)
					if current_step % FLAGS.evaluate_every == 0:
							print("\nEvaluation:")
							validation_step(x_validation_front, x_validation_behind, y_validation, writer=validation_summary_writer)
							print("")
					if current_step % FLAGS.checkpoint_every == 0:
							path = saver.save(sess, checkpoint_prefix, global_step=current_step)
							print("Saved model checkpoint to {}\n".format(path))

			# Saving graph
			print("Saving graph...")
			tf.train.write_graph(sess.graph, checkpoint_dir, FLAGS.input_graph_name)

			# exporting graph and model
			print("Freezing model...")
			input_graph_path = os.path.join(checkpoint_dir, FLAGS.input_graph_name)
			output_graph_path = os.path.join(checkpoint_dir, FLAGS.output_graph_name)
			model_exports.freeze_model(input_graph_path, output_graph_path, FLAGS.output_node, path)

main()

