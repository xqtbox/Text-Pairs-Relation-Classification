# -*- coding:utf-8 -*-

import os
import logging
import numpy as np
import tensorflow as tf
import data_helpers

logging.getLogger().setLevel(logging.INFO)


user_input = input("☛ Please input the subset and the model file you want to test, it should be like(11, 1490175368): ")
SUBSET = user_input.split(',')[0]
MODEL_LOG = user_input.split(',')[1][1:]

while not (SUBSET.isdigit() and int(SUBSET) in range(1, 12) and MODEL_LOG.isdigit() and len(MODEL_LOG) == 10):
    SUBSET = input('✘ The format of your input is illegal, it should be like(11, 1490175368), please re-input: ')
logging.info('✔︎ The format of your input is legal, now loading to next step...')

SAVE_FILE = 'result' + SUBSET + '.txt'

BASE_DIR = os.getcwd()
VALIDATIONSET_DIR = BASE_DIR + '/Model Validation' + '/Model' + SUBSET + '_Validation.txt'
TESTSET_DIR = BASE_DIR + '/Model Test' + '/Model' + SUBSET + '_Test.txt'
MODEL_DIR = BASE_DIR + '/runs/' + MODEL_LOG + '/checkpoints/'
MODEL_FILE = BASE_DIR + '/runs/' + MODEL_LOG + '/checkpoints/output_graph.pb'

FLAGS = tf.flags.FLAGS

# Data loading params
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data")
tf.flags.DEFINE_string("test_data_file", TESTSET_DIR, "Data source for the test data")
tf.flags.DEFINE_string("checkpoint_dir", MODEL_DIR, "Checkpoint directory from training run")

# Data parameters
tf.flags.DEFINE_string("MAX_SEQUENCE_LENGTH", 550, "每个文本的最长选取长度(padding的统一长度),较短的文本可以设短些.")
tf.flags.DEFINE_string("MAX_NB_WORDS", 10000, "整体词库字典中，词的多少，可以略微调大或调小.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
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
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

# Model export parameters
tf.flags.DEFINE_string("input_graph_name", "input_graph.pb", "Graph input file of the graph to export")
tf.flags.DEFINE_string("output_graph_name", "output_graph.pb", "Graph output file of the graph to export")
tf.flags.DEFINE_string("output_node", "output/predictions", "The output node of the graph")


def test_cnn():
    """Test CNN model."""

    # Load data
    logging.info("✔ Loading data...")
    x_test_front, x_test_behind, y_test = \
        data_helpers.load_data_and_labels(FLAGS.validation_data_file, FLAGS.MAX_SEQUENCE_LENGTH, FLAGS.embedding_dim)

    vocab_size = data_helpers.load_vocab_size()
    pretrained_word2vec_matrix = data_helpers.load_word2vec_matrix(vocab_size, FLAGS.embedding_dim)

    logging.info("✔ Loading model...")
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logging.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
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
            sigmoidScores = graph.get_operation_by_name("output/sigmoidScores").outputs[0]
            topKPreds = graph.get_operation_by_name("output/topKPreds").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(zip(x_test_front, x_test_behind)), FLAGS.batch_size, 1,
                                              shuffle=False)

            # Collect the predictions here
            all_scores = []
            all_softMaxScores = []
            all_sigmoidScores = []
            all_predictions = []
            all_topKPreds = []

            for x_test_batch in batches:
                x_batch_front, x_batch_behind = zip(*x_test_batch)
                feed_dict = {
                    input_x_front: x_batch_front,
                    input_x_behind: x_batch_behind,
                    dropout_keep_prob: 1.0
                }
                batch_scores = sess.run(scores, feed_dict)
                all_scores = np.append(all_scores, batch_scores)

                batch_softMax_scores = sess.run(softmaxScores, feed_dict)
                all_softMaxScores = np.append(all_softMaxScores, batch_softMax_scores)

                batch_sigmoid_Scores = sess.run(sigmoidScores, feed_dict)
                all_sigmoidScores = np.append(all_sigmoidScores, batch_sigmoid_Scores)

                batch_predictions = sess.run(predictions, feed_dict)
                all_predictions = np.concatenate([all_predictions, batch_predictions])

                batch_topKPreds = sess.run(topKPreds, feed_dict)
                all_topKPreds = np.append(all_topKPreds, batch_topKPreds)

            np.savetxt(SAVE_FILE, list(zip(all_predictions, all_topKPreds)), fmt='%s')

    logging.info("✔ Done.")


if __name__ == '__main__':
    test_cnn()
