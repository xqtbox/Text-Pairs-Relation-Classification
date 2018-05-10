# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import tensorflow as tf


def linear(input_, output_size, scope=None):
    """
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        input_: a tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: {0}".format(str(shape)))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: {0}".format(str(shape)))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        W = tf.get_variable("W", [output_size, input_size], dtype=input_.dtype)
        b = tf.get_variable("b", [output_size], dtype=input_.dtype)

    return tf.nn.xw_plus_b(input_, tf.transpose(W), b)


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """
    Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope=('highway_lin_{0}'.format(idx))))
            t = tf.sigmoid(linear(input_, size, scope=('highway_gate_{0}'.format(idx))) + bias)
            output = t * g + (1. - t) * input_
            input_ = output

    return output


class TextCNN(object):
    """A CNN for text classification."""

    def __init__(
            self, sequence_length, num_classes, vocab_size, fc_hidden_size, embedding_size,
            embedding_type, filter_sizes, num_filters, l2_reg_lambda=0.0, pretrained_embedding=None):

        # Placeholders for input, output, dropout_prob and training_tag
        self.input_x_front = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_front")
        self.input_x_behind = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_behind")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        # Embedding Layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # Use random generated the word vector by default
            # Can also be obtained through our own word vectors trained by our corpus
            if pretrained_embedding is None:
                self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0,
                                                               dtype=tf.float32), trainable=True, name="embedding")
            else:
                if embedding_type == 0:
                    self.embedding = tf.constant(pretrained_embedding, dtype=tf.float32, name="embedding")
                if embedding_type == 1:
                    self.embedding = tf.Variable(pretrained_embedding, trainable=True,
                                                 dtype=tf.float32, name="embedding")
            self.embedded_sentence_front = tf.nn.embedding_lookup(self.embedding, self.input_x_front)
            self.embedded_sentence_behind = tf.nn.embedding_lookup(self.embedding, self.input_x_behind)
            self.embedded_sentence_expanded_front = tf.expand_dims(self.embedded_sentence_front, -1)
            self.embedded_sentence_expanded_behind = tf.expand_dims(self.embedded_sentence_behind, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_front = []
        pooled_outputs_behind = []

        for filter_size in filter_sizes:
            with tf.name_scope("conv-filter{0}".format(filter_size)):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1, dtype=tf.float32), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters], dtype=tf.float32), name="b")
                conv_front = tf.nn.conv2d(
                    self.embedded_sentence_expanded_front,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_front")

                conv_behind = tf.nn.conv2d(
                    self.embedded_sentence_expanded_behind,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_behind")

                # Batch Normalization Layer
                conv_bn_front = tf.layers.batch_normalization(tf.nn.bias_add(conv_front, b), training=self.is_training)
                conv_bn_behind = tf.layers.batch_normalization(tf.nn.bias_add(conv_behind, b), training=self.is_training)

                # Apply nonlinearity
                conv_out_front = tf.nn.relu(conv_bn_front, name="relu_front")
                conv_out_behind = tf.nn.relu(conv_bn_behind, name="relu_behind")

            with tf.name_scope("pool-filter{0}".format(filter_size)):
                # Maxpooling over the outputs
                pooled_front = tf.nn.max_pool(
                    conv_out_front,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool_front")

                pooled_behind = tf.nn.max_pool(
                    conv_out_behind,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool_behind")

            pooled_outputs_front.append(pooled_front)
            pooled_outputs_behind.append(pooled_behind)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.pool_front = tf.concat(pooled_outputs_front, 3)
        self.pool_behind = tf.concat(pooled_outputs_behind, 3)
        self.pool_flat_front = tf.reshape(self.pool_front, [-1, num_filters_total])
        self.pool_flat_behind = tf.reshape(self.pool_behind, [-1, num_filters_total])

        self.pool_flat_combine = tf.concat([self.pool_flat_front, self.pool_flat_behind], 1)

        # Fully Connected Layer
        with tf.name_scope("fc"):
            W = tf.Variable(tf.truncated_normal(shape=[num_filters_total * 2, fc_hidden_size],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[fc_hidden_size], dtype=tf.float32), name="b")
            self.fc = tf.nn.xw_plus_b(self.pool_flat_combine, W, b)

            # Batch Normalization Layer
            self.fc_bn = tf.layers.batch_normalization(self.fc, training=self.is_training)

            # Apply nonlinearity
            self.fc_out = tf.nn.relu(self.fc_bn, name="relu")

        # Highway Layer
        self.highway = highway(self.fc_out, self.fc_out.get_shape()[1], num_layers=1, bias=0, scope="Highway")

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.highway, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, num_classes],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes], dtype=tf.float32), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.softmax_scores = tf.nn.softmax(self.logits, name="softmax_scores")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            self.topKPreds = tf.nn.top_k(self.softmax_scores, k=1, sorted=True, name="topKPreds")

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_mean(losses, name="softmax_losses")
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * l2_reg_lambda
            self.loss = tf.add(losses, l2_losses, name="loss")

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Number of correct predictions
        with tf.name_scope("num_correct"):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, "float"), name="num_correct")

        # Calculate Tp
        with tf.name_scope("tp"):
            tp = tf.metrics.true_positives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions)
            self.tp = tf.reduce_sum(tf.cast(tp, "float"), name="tp")


        # Calculate Fp
        with tf.name_scope("fp"):
            fp = tf.metrics.false_positives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions)
            self.fp = tf.reduce_sum(tf.cast(fp, "float"), name="fp")

        # Calculate Fn
        with tf.name_scope("fn"):
            fn = tf.metrics.false_negatives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions)
            self.fn = tf.reduce_sum(tf.cast(fn, "float"), name="fn")

        # Calculate Precision
        with tf.name_scope("precision"):
            # self.precision = self.num_correct / (self.num_correct + self.fp)
            self.precision = self.tp / (self.tp + self.fp)

            # Calculate Recall
        with tf.name_scope("recall"):
            # self.recall = self.num_correct / (self.num_correct + self.fn)
            self.recall = self.tp / (self.tp + self.fn)

        # Calculate F1
        with tf.name_scope("F1"):
            self.F1 = tf.add((2 * self.precision * self.recall) / (self.precision + self.recall), 0.00,name="F1")

        # Calculate AUC
        with tf.name_scope("AUC"):
            self.AUC = tf.metrics.auc(labels=tf.argmax(self.input_y, 1), predictions=self.predictions, name="AUC")