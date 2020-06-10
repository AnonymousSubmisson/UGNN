# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


# def _norm(input_, is_training, scope):
#     # return tf.keras.activations.relu(tf.keras.layers.BatchNormalization()(input_))
#     return tf.nn.relu(tf.contrib.layers.batch_norm(input_, is_training=is_training, scope=scope))


def _sat_linear(input_, nin, nout, scope=None):
    with tf.variable_scope(scope, 'sat_linear', initializer=tf.truncated_normal_initializer()):
        weight = tf.get_variable('weight', shape=(nin, nout), dtype=tf.float32)
        bias = tf.get_variable('bias', shape=(
            nout), dtype=tf.float32, initializer=tf.zeros_initializer())
        return tf.nn.bias_add(tf.matmul(input_, weight), bias)


def sat_infer(num_vars, num_clauses, edges, dropout_rate, lits_index, clauses_index, name=None):

    OUTPUT_ND = 64
    EMBEDDING_SIZE_VAR = 100
    EMBEDDING_SIZE_CLAUSE = 300

    keep_probe = 1 - dropout_rate

    with tf.variable_scope(name, 'sat_infer'):

        USE_DIFFERENT_EMBEDDING = True

        num_vars_32 = tf.cast(num_vars, tf.int32)
        num_clauses_32 = tf.cast(num_clauses, tf.int32)

        if USE_DIFFERENT_EMBEDDING:
            # lits = tf.range(0, num_vars_32*2, 1) % EMBEDDING_SIZE_VAR
            # clauses = tf.range(0, num_clauses_32, 1) % EMBEDDING_SIZE_CLAUSE
            lits_embedding1 = tf.get_variable('lits1', shape=(EMBEDDING_SIZE_VAR, OUTPUT_ND),  initializer=tf.random_normal_initializer())
            lits_embedding2 = tf.get_variable('lits2', shape=(EMBEDDING_SIZE_VAR, OUTPUT_ND),  initializer=tf.random_normal_initializer())
            clauses_embedding = tf.get_variable('clauses', shape=(EMBEDDING_SIZE_CLAUSE, OUTPUT_ND),  initializer=tf.random_normal_initializer())
            lits1 = tf.nn.embedding_lookup(lits_embedding1, lits_index)
            lits2 = tf.nn.embedding_lookup(lits_embedding2, lits_index)
            lits = tf.concat([lits1, lits2], axis=0)
            clauses = tf.nn.embedding_lookup(clauses_embedding, clauses_index)
        else:
            lits = tf.get_variable('lits', shape=(
                1, OUTPUT_ND), initializer=tf.random_normal_initializer())
            clauses = tf.get_variable('clauses', shape=(
                1, OUTPUT_ND), initializer=tf.random_normal_initializer())
            lits = tf.tile(lits, tf.convert_to_tensor([num_vars_32*2, 1]))
            clauses = tf.tile(clauses, tf.convert_to_tensor([num_clauses_32, 1]))

        lits = tf.contrib.rnn.LSTMStateTuple(h=lits, c=tf.zeros(
            tf.convert_to_tensor([num_vars_32*2, OUTPUT_ND], dtype=tf.int32)))
        clauses = tf.contrib.rnn.LSTMStateTuple(h=clauses, c=tf.zeros(
            tf.convert_to_tensor([num_clauses_32, OUTPUT_ND], dtype=tf.int32)))

        # lits_layer = tf.contrib.rnn.LayerNormBasicLSTMCell(OUTPUT_ND)
        # clauses_layer = tf.contrib.rnn.LayerNormBasicLSTMCell(OUTPUT_ND)

        for i in range(16):
            with tf.variable_scope(tf.get_variable_scope(), reuse=None  # if i == 0 else True
                                   ):
                with tf.variable_scope('layer_%d' % (i,)):
                    lits_msg = tf.nn.relu(_sat_linear(
                        lits.h, OUTPUT_ND, OUTPUT_ND, "lits_msg_1"), name="lits_msg_1_relu")
                    # if is_training:
                    lits_msg = tf.nn.dropout(lits_msg, keep_probe)
                    lits_msg = _sat_linear(
                        lits_msg, OUTPUT_ND, OUTPUT_ND, "lits_msg_3")
                    lits_msg = tf.sparse_tensor_dense_matmul(edges, lits_msg)
                    _, clauses = tf.contrib.rnn.LayerNormBasicLSTMCell(OUTPUT_ND)(
                        inputs=lits_msg, state=clauses, scope='clauses_layer')
                    clauses_msg = tf.nn.relu(_sat_linear(
                        clauses.h, OUTPUT_ND, OUTPUT_ND, "clauses_msg_1"), name="clauses_msg_1_relu")
                    # if is_training:
                    clauses_msg = tf.nn.dropout(clauses_msg, keep_probe)
                    clauses_msg = _sat_linear(
                        clauses_msg, OUTPUT_ND, OUTPUT_ND, "clauses_msg_3")
                    clauses_msg = tf.sparse_tensor_dense_matmul(
                        edges, clauses_msg, adjoint_a=True)

                    flipinfo = tf.concat(
                        [lits.h[num_vars_32: num_vars_32*2], lits.h[0:num_vars_32, :]], axis=0)
                    allinfo = tf.concat([clauses_msg, flipinfo], axis=1)

                    _, lits = tf.contrib.rnn.LayerNormBasicLSTMCell(OUTPUT_ND)(
                        inputs=allinfo, state=lits, scope='lits_layer')

        varoutput = tf.concat([
            lits.h[0:num_vars_32, :], lits.h[num_vars_32: num_vars_32*2]
        ], axis=1)

        varoutput = tf.nn.relu(_sat_linear(varoutput, OUTPUT_ND * 2, 512, "out_linear_1"))
        # if is_training:
        varoutput = tf.nn.dropout(varoutput, keep_probe)
        #varoutput = tf.nn.leaky_relu(_sat_linear(
        #    varoutput, 64, 8, "output_linear_4"), name="output_linear_4_relu")
        # if is_training:
        #varoutput = tf.nn.dropout(varoutput, keep_probe)
        varoutput = _sat_linear(varoutput, 512, 2, "output_linear_5")

        return varoutput


def sat_loss(infer, output_labels, name=None):
    """SAT loss

    Arguments:
        infer {Tensor(shape=(num_nodes, 2))} -- infer
        output_labels {Tensor(shape=(num_nodes, 2))} -- outputlabels
        output_weights {Tensor(shape=(num_nodes))} -- weights

    Returns:
        Tensor(shape=()) -- loss
    """

    with tf.variable_scope(name, 'sat_loss'):
        # infer = tf.Print(infer, [infer], summarize=100, message='infer')
        # output_labels = tf.Print(output_labels, [output_labels], summarize=100, message='ol')
        # output_weights = tf.Print(output_weights, [output_weights], summarize=100, message='ow')
        # loss = tf.nn.softmax_cross_entropy_with_logits(
        #     logits=infer, labels=output_labels)
        # loss = loss * output_weights
        # loss = tf.reduce_sum(loss) / (tf.reduce_sum(output_weights) + 1e-5)
        # return loss
        # return tf.losses.softmax_cross_entropy(onehot_labels=output_labels, logits=infer, weights=output_weights) / tf.reduce_sum(output_weights)
        # return tf.losses.sigmoid_cross_entropy(infer[:, 0], output_labels[:, 0], output_weights) / tf.reduce_sum(output_weights)
        output_labels = tf.one_hot(output_labels, 2)
        return tf.losses.softmax_cross_entropy(onehot_labels=output_labels, logits=infer)
