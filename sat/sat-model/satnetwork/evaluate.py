
from .network import sat_infer, sat_loss

import multiprocessing
import random
import time
import sys
import os
import struct
import gzip
import pickle
import tqdm

import numpy as np
import tensorflow as tf


class SATModel:
    def __init__(self, config):
        self.config = config

        self.pre = 0
        self._build_network()
        self.global_step = tf.get_variable(
            'global_step', initializer=tf.constant_initializer(0, dtype=tf.int32), shape=(), trainable=False)
        self.increase_global_step = tf.assign_add(self.global_step, 1)

        self.saver = tf.train.Saver(
            tf.trainable_variables() + [self.global_step])
        if not self.config.is_evaluate:
            self.writer = tf.summary.FileWriter(os.path.join(self.config.logdir, 'train'))
            self.writer_eval = tf.summary.FileWriter(os.path.join(self.config.logdir, 'eval'))
        tfconfig = tf.ConfigProto()
        tfconfig.allow_soft_placement = True
        tfconfig.gpu_options.allow_growth = True # pylint: disable=E1101
        self.sess = tf.Session(
            config=tfconfig)
        self.sess.run(tf.global_variables_initializer())
        if not self.config.is_evaluate:
            self.writer.add_graph(self.sess.graph)

        self.train_data = iter(self._fetch_train_data())
        self.eval_data = iter(self._fetch_eval_data())

    def random_index(self, index, num):
        sum = 0
        start = 0
        end = 0
        ans = []
        l = []
        #print ("shape", np.shape(index)) 
        import random
        num = [num]
        #for n in num:
        n = num[0]
        sum += n
        end += n
            #for i in range(start, end):
            #    l.append()
        random_index = np.random.permutation(n)
            
            #random.shuffle(l)
            #ans += l
        new_index = index[random_index,]#np.array(ans)
        #assert len(index) == sum
        return new_index

    def _fetch_train_data(self):
        while True:
            files = os.listdir(self.config.train_data)
            random.shuffle(files)
            for f in files:
                if f <= '000000c7.gz':
                    continue
                try:
                    finput = gzip.open(os.path.join(self.config.train_data, f))
        
                    num_vars, num_clauses, labels, edges, lits_index, clauses_index = pickle.load(finput)
                    yield {
                        self.input_num_vars: num_vars, 
                        self.input_num_clauses: num_clauses, 
                        self.input_labels: labels,
                        self.input_edges: edges,
                        self.lits_index: self.random_index(lits_index, num_vars),
                        self.clauses_index: self.random_index(clauses_index, num_clauses)
                    }
                except Exception as e:
                    print(f'Load ERROR {f}: {e}')

    def _fetch_eval_data(self):
        while True:
            files = os.listdir(self.config.eval_data)
            for f in files:
                try:
                    finput = gzip.open(os.path.join(self.config.eval_data, f))
                    num_vars, num_clauses, labels, edges, lits_index, clauses_index = pickle.load(finput)
                    yield {
                        self.input_num_vars: num_vars, 
                        self.input_num_clauses: num_clauses, 
                        self.input_labels: labels,
                        self.input_edges: edges,
                        self.lits_index: lits_index,
                        self.clauses_index: clauses_index
                    }
                except Exception as e:
                    print(f'Load ERROR {f}: {e}')

    def _build_network(self):
        with tf.name_scope('train_data'):
            num_vars = self.input_num_vars = tf.placeholder(tf.int64, shape=[], name='input_num_vars')
            num_clauses = self.input_num_clauses = tf.placeholder(tf.int64, shape=[], name='input_num_clauses')
            labels = self.input_labels = tf.placeholder(tf.int64, shape=[None], name='input_labels')
            edges = self.input_edges = tf.placeholder(tf.int64, shape=[None, 2], name='input_edges')

            lits_index = self.lits_index = tf.placeholder(tf.int64, shape=[None], name='lits_index')
            clauses_index = self.clauses_index = tf.placeholder(tf.int64, shape=[None], name='clauses_index')

            edges = tf.SparseTensor(indices=edges, values=tf.ones(
                shape=[tf.shape(edges)[0]]), dense_shape=[num_clauses, num_vars*2])
            edges = tf.sparse.reorder(edges)
            sum0 = tf.math.sqrt(tf.sparse.reduce_sum(edges, 0, keepdims=True)) + 1e-6
            sum1 = tf.math.sqrt(tf.sparse.reduce_sum(edges, 1, keepdims=True)) + 1e-6
            edges = edges / sum0 / sum1
        with tf.name_scope('train_infer'):
            self.dropout_rate = tf.get_variable(
                'dropout_rate', initializer=tf.constant_initializer(0, dtype=tf.float32), shape=(), trainable=False)
            self.infer = infer = sat_infer(
                num_vars, num_clauses, edges, self.dropout_rate, lits_index, clauses_index)
            self.softmax_infer = tf.nn.softmax(infer, -1)
        with tf.name_scope('train_loss'):
            self.loss = loss = sat_loss(infer, labels)
        if not self.config.is_evaluate:
            with tf.name_scope('train_optimize'):
                optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
                grads = optimizer.compute_gradients(
                    loss, colocate_gradients_with_ops=True)
                grads = [(tf.clip_by_norm(grad, self.config.clip_norm), var)
                        for grad, var in grads]
                self.optimize = optimizer.apply_gradients(grads)
            with tf.name_scope('train_summary'):
                self.summary_ph = tf.placeholder(dtype=tf.float32, shape=[])
                self.summary = tf.summary.scalar('loss', self.summary_ph)
                self.summary_percision = tf.summary.scalar('percision', self.summary_ph)
                histogram_list = []
                for var in tf.trainable_variables():
                    histogram_list.append(tf.summary.histogram(var.name, var))
                self.histogramsum = tf.summary.merge(histogram_list)
    
    def train(self, steps):
        while steps > 0:
            self.change_dropout(0.1)
            for _ in tqdm.tqdm(list(range(self.config.train_length))):
                _, loss, _, step = self.sess.run([
                    self.optimize,
                    self.loss,
                    self.increase_global_step,
                    self.global_step],
                    feed_dict=next(self.train_data))
                summary = self.sess.run(self.summary, {self.summary_ph: loss})
                self.writer.add_summary(summary, step)
                steps -= 1
                if steps <= 0:
                    break
            histogramsum = self.sess.run(self.histogramsum)
            self.writer.add_summary(histogramsum, step)
            self.saver.save(self.sess, os.path.join(
                self.config.logdir, 'debug'), global_step=self.global_step)
            self.writer.flush()
            self.evaluate(self.config.eval_data_size)

    def random_shuffle(self, array):
        last_end = 0
        array = np.array(array)
        change = np.arange(0, len(array), 1, dtype=np.int32)
        array = np.concatenate([np.expand_dims(array, 1), np.expand_dims(change, 1)], axis=1)
        for i in range(len(array)):
            if i != 0 and array[i][0] == 0:
                np.random.shuffle(array[last_end: i])
                last_end = i
        np.random.shuffle(array[last_end:])
        return array[:,0], array[:,1]

    def gen_eval_set(self, eval_data):
        eval_data = dict(eval_data.items())
        eval_data[self.lits_index], lit_inv = self.random_shuffle(eval_data[self.lits_index])
        eval_data[self.clauses_index], cla_inv = self.random_shuffle(eval_data[self.clauses_index])
        return eval_data, lit_inv, cla_inv


    def evaluate(self, steps):
        self.change_dropout(0)

        NUM_TREES = 20

        totalloss = 0
        correct = 0
        total = 0
        
        f_infer = ""
        for NUM_TREES in tqdm.tqdm(range(1, 51)):
            totalloss = 0
            correct = 0
            total = 0
            for _ in tqdm.tqdm(list(range(steps))):
                eval_data = next(self.eval_data)

                num_vars = eval_data[self.input_num_vars]
                total_infer = np.zeros(shape=(num_vars, 2))

                f_infer = ""
                for tr in range(NUM_TREES):
                    eval_data_new, lit_inv, cla_inv = self.gen_eval_set(eval_data)
                    # eval_data_new = eval_data

                    infer, labels, num_nodes, loss = self.sess.run([
                        self.softmax_infer,
                        self.input_labels,
                        self.input_num_vars,
                        self.loss],
                        feed_dict=eval_data_new)
                    totalloss += loss / NUM_TREES
                    if f_infer == "": 
                        f_infer = infer
                    else:
#                        b_infer = deepcopy(f_infer)
                        b_infer = infer#deepcopy(infer)
                        #b_infer = np.clip(np.exp(b_infer), 1e-9, 1e9)
                        #s_b_infer = b_infer.sum(axis=-1, keepdims=True)
                        #b_infer /= s_b_infer
                        f_infer = np.maximum(f_infer, b_infer)
                    total_infer = np.argmax(f_infer, 1)
                    open(str(tr) + "-5.pkl", "wb").write(pickle.dumps(total_infer == labels))
#                    infer = np.argmax(infer, -1)
                    # infer[lit_inv] = infer
#                    total_infer[np.arange(num_vars), infer] += 1
                total_infer = np.argmax(f_infer, 1)
                correct += int(np.sum(total_infer == labels))
                total += int(num_nodes)


            totalloss /= steps

            if self.config.is_evaluate:
                print(f'percision: {correct / total}')
            else:
                summary = self.sess.run(self.summary, {self.summary_ph: totalloss})
                self.writer_eval.add_summary(summary, self.sess.run(self.global_step))
                summary = self.sess.run(self.summary_percision, {self.summary_ph: correct / total})
                self.writer_eval.add_summary(summary, self.sess.run(self.global_step))
                self.writer_eval.flush()
                if correct / total > self.pre:
                    self.pre = correct / total
                    self.saver.save(self.sess, os.path.join(
                        self.config.logdir, 'best'), global_step=self.global_step)


    def load_model(self, file):
        self.saver.restore(self.sess, file)

    def change_dropout(self, val):
        self.sess.run(tf.assign(self.dropout_rate, val))

    def run_predict(self, num_vars, num_clauses, edges):
        self.change_dropout(0)
        infer = self.sess.run(
            self.infer,
            feed_dict={
                self.input_num_vars: num_vars,
                self.input_num_clauses: num_clauses,
                self.input_edges: edges})
        infer = np.argmax(infer, -1).astype(np.int32)
        return infer
