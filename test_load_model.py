import randolph
import model_exports
import tensorflow as tf
import datetime
import numpy as np
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

BASE_DIR = randolph.cur_file_dir()
model_file = BASE_DIR + "/runs/1489987989/checkpoints/output_graph.pb"

# Data loading params
tf.flags.DEFINE_string("training_data_file", BASE_DIR + '/Model1_Training.txt', "Data source for the training data.")
tf.flags.DEFINE_string("test_data_file", BASE_DIR + '/Model1_Test.txt', "Data source for the test data.")
tf.flags.DEFINE_string("checkpoint_dir", BASE_DIR + '/runs/1489987989/checkpoints', "Checkpoint directory from training run")

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
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Model export parameters
tf.flags.DEFINE_string("input_graph_name", "input_graph.pb", "Graph input file of the graph to export")
tf.flags.DEFINE_string("output_graph_name", "output_graph.pb", "Graph output file of the graph to export")
tf.flags.DEFINE_string("output_node", "output/predictions", "The output node of the graph")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_test_front, x_test_behind, y_test = \
    data_helpers.load_data_and_labels(FLAGS.test_data_file, FLAGS.MAX_SEQUENCE_LENGTH, FLAGS.embedding_dim)

vocab_size = data_helpers.load_vocab_size()
pretrained_word2vec_matrix = data_helpers.load_word2vec_matrix(vocab_size, FLAGS.embedding_dim)

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(config=session_conf)

	with sess.as_default():
		# Load the saved meta graph and restore variables
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)

		# Get the placeholders from the graph by name
		input_x_front = graph.get_operation_by_name("input_x_front").outputs[0]
		input_x_behind = graph.get_operation_by_name("input_x_behind").outputs[0]

		# input_y = graph.get_operation_by_name("input_y").outputs[0]
		dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

		# pre-trained_word2vec
		pretrained_embedding = graph.get_operation_by_name("embedding/W").outputs[0]

		# Tensors we want to evaluate
		scores = graph.get_operation_by_name("output/scores").outputs
		predictions = graph.get_operation_by_name("output/predictions").outputs[0]
		softmaxScores = graph.get_operation_by_name("output/softmaxScores").outputs[0]
		topKPreds = graph.get_operation_by_name("output/topKPreds").outputs[0]

		# Generate batches for one epoch
		# batches = data_helpers.batch_iter(
		# 	list(zip(x_test_front, x_test_behind)), FLAGS.batch_size, 1)

		# Collect the predictions here
		all_scores = []
		all_softMaxScores = []
		all_predictions = []
		all_topKPreds = []

		feed_dict = {
			input_x_front: x_test_front,
			input_x_behind: x_test_behind,
			dropout_keep_prob: 1.0
		}

		batch_scores = sess.run(scores, feed_dict)
		all_scores = np.append(all_scores, batch_scores)

		batch_softMax_scores = sess.run(softmaxScores, feed_dict)
		all_softMaxScores = np.append(all_softMaxScores, batch_softMax_scores)

		batch_predictions = sess.run(predictions, feed_dict)
		all_predictions = np.concatenate([all_predictions, batch_predictions])

		batch_topKPreds = sess.run(topKPreds, feed_dict)
		all_topKPreds = np.append(all_topKPreds, batch_topKPreds)

		# all_softMaxScores    = np.append(all_softMaxScores, [[smxScore] for smxScore in batch_softMax_scores])

print(all_scores)
print(all_predictions[17])
print(all_topKPreds[17])

np.savetxt('result.txt', list(zip(all_scores, all_predictions, all_topKPreds)), fmt='%s')
# with open('result.txt', 'w') as fout:
# 	for i in range(len(x_test_front)):
# 		outStr = ''
# 		outStr = outStr + str(i) + '\t' + all_predictions[i].astype('S32')
# 		fout.write(outStr + '\n')
# saver = tf.train.import_meta_graph(cpkl.model_checkpoint_path + '.meta')
# saver.restore(sess, cpkl.model_checkpoint_path)
#
# dp_dict = tl.utils.dict_to_one(cnn.outputnetwork.all_drop)
# feed_dict = {
#     cnn.input_x_front: x_test_front,
#     cnn.input_x_behind: x_test_behind,
#     cnn.input_y: y_test,
#     cnn.dropout_keep_prob: 1.0
# }
# feed_dict.update(dp_dict)
#
# print(sess.run(cnn.accuracy, feed_dict=feed_dict))


# # We use our load_graph function on the file
# graph = model_export.load_model(model_file)
# print(graph)
#
# # We can verify that we can access to the list of operations in the graph
# for op in graph.get_operations():
#     print(op.name)
#
# # We access the input and output nodes
# input_x_front = graph.get_tensor_by_name('prefix/input_x_front:0')
# input_x_behind = graph.get_tensor_by_name('prefix/input_x_behind:0')
# scores = graph.get_tensor_by_name('prefix/output/scores:0')
# predictions = graph.get_tensor_by_name('prefix/output/predictions:0')
# drop = graph.get_tensor_by_name('prefix/dropout_keep_prob:0')
#
# # We launch a Session
# with tf.Session(graph=graph) as sess:
#     input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
#     # CalculateMean cross-entropy loss
#     with tf.name_scope("loss"):
#         losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
#         loss = tf.reduce_mean(losses)
#
#     # Accuracy
#     with tf.name_scope("accuracy"):
#         correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
#
#     # AUC
#     with tf.name_scope("AUC"):
#         AUC = tf.contrib.metrics.streaming_auc(predictions, tf.argmax(input_y, 1))
        
    # example = [0] * 56
    # example[0] = 7080
    # example[1] = 2294
    # example[2] = 1776
    # example[3] = 2344
    # example[4] = 14041
    # example[5] = 941
    # example[6] = 12
    # example[7] = 8186
    # example[8] = 1991
    #
    # print(example)
    # example = [example]
    # print(example)
    #
    # print(predictions.eval(feed_dict = { input_x : example, drop:1.0}))
    #
    