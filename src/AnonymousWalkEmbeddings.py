from __future__ import division, print_function

import os
import math
import random
import argparse
import sys
import time
import re
import shutil
import threading
from collections import Counter

import numpy as np
import torch

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42

class AWE(object):
    '''
    Computes distributed Anonymous Walk Embeddings.
    '''
    def __init__(self,
                 dataset='imdb_b',
                 batch_size=128,
                 window_size=8,
                 concat=False,
                 embedding_size_w=64,
                 embedding_size_d=64,
                 loss_type='sampled_softmax',
                 num_samples=64,
                 optimize='Adagrad',
                 learning_rate=1.0,
                 root = '../',
                 ext = 'graphml',
                 steps = 7,
                 epochs = 1,
                 batches_per_epoch = 1,
                 candidate_func = None,
                 graph_labels = None,
                 regenerate_corpus = False,
                 neighborhood_size=1):
        '''
        Initialize AWE model.
        :param dataset: name of the dataset and corresponding name of the folder.
        :param batch_size: number of batches per iteration of AWE model.
        :param window_size: number of context words.
        :param concat: Concatenate context words or not.
        :param embedding_size_w: embedding size of word
        :param embedding_size_d: embedding size of document
        :param loss_type: sampled softmax or nce
        :param num_samples: number of (negative) samples for every target word.
        :param optimize: SGD or Adagrad
        :param learning_rate: learning rate of the model
        :param root: root folder of the dataset
        :param ext: extension of files with graphs (e.g. graphml)
        :param steps: length of anonymous walk
        :param epochs: number of epochs for iterations
        :param batches_per_epoch: number of batches per epoch for each graph
        :param candidate_func: None (loguniform by default) or uniform
        :param graph_labels: None, edges, nodes, edges_nodes
        '''

        # bind params to class
        self.batch_size = batch_size
        self.window_size = window_size
        self.concat = concat
        self.embedding_size_w = embedding_size_w
        self.embedding_size_d = embedding_size_d
        self.loss_type = loss_type
        self.num_samples = num_samples
        self.optimize = optimize
        self.learning_rate = learning_rate
        self.candidate_func = candidate_func
        self.graph_labels = graph_labels

        self.ROOT = root
        self.ext = ext
        self.steps = steps
        self.epochs = epochs
        self.dataset = dataset

        self.batches_per_epoch = batches_per_epoch


        # switch to have batches_per_epoch = N for every graph with N nodes
        self.flag2iterations = False
        if batches_per_epoch is None:
            self.flag2iterations = True

        # get all graph filenames (document size)
        self.folder = self.ROOT + self.dataset + '/'
        folder_graphs = filter(lambda g: g.endswith(max(self.ext, '')), os.listdir(self.folder))

        self.sorted_graphs = sorted(folder_graphs, key=lambda g: int(re.findall(r'\d+', g)[0]))
        self.document_size = len(self.sorted_graphs)
        print('Number of graphs: {}'.format(self.document_size))

        print('Generating corpus... ', end='')
        self.corpus_fn_name = '{}.corpus'
        self.regenerate_corpus = regenerate_corpus
        self.neiborhood_size = neighborhood_size
        start2gen = time.time()
        self.generate_corpus()
        print('Finished {}'.format(time.time() - start2gen))

        self.vocabulary_size = max(self.walk_ids.values()) + 1
        print('Number of words: {}'.format(self.vocabulary_size))

        # init all variables in a tensorflow graph
        self._init_graph()

        # create a session
        self.sess = tf.Session(graph=self.graph)

    def generate_corpus(self):
        # get all AW (vocabulary size)
        self.g2v = AnonymousWalks()
        if self.graph_labels is None:
            self.g2v._all_paths(self.steps, keep_last=True)
        elif self.graph_labels == 'nodes':
            self.g2v._all_paths_nodes(self.steps, keep_last=True)
        elif self.graph_labels == 'edges':
            self.g2v._all_paths_edges(self.steps, keep_last=True)
        elif self.graph_labels == 'edges_nodes':
            self.g2v._all_paths_edges_nodes(self.steps, keep_last=True)

        self.walk_ids = dict()
        for i, path in enumerate(self.g2v.paths[self.steps]):
            self.walk_ids[tuple(path)] = i

        self.nodes_per_graphs = dict()

        label_suffix = ''
        if graph_labels is not None:
            label_suffix = '_' + graph_labels

        if self.regenerate_corpus == True or not os.path.exists(self.ROOT + self.dataset + '_corpus' + label_suffix):
            if not os.path.exists(self.ROOT + self.dataset + '_corpus' + label_suffix):
                os.mkdir(self.ROOT + self.dataset + '_corpus' + label_suffix)

            for en, graph_fn in enumerate(self.sorted_graphs):
                if en > 0 and not en%100:
                    print(f"Graph {en}")
                g2v = AnonymousWalks()
                g2v.read_graphml(self.folder + graph_fn)
                self.nodes_per_graphs[en] = len(g2v.graph)


                g2v.write_corpus(self.neiborhood_size, self.walk_ids, steps, self.graph_labels,
                                 self.ROOT + self.dataset + '_corpus{}/{}'.format(label_suffix, self.corpus_fn_name.format(en)))

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing:
        input data, variables, model, loss function, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            tf.set_random_seed(SEED)

            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size+1])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            # embeddings for anonymous walks
            self.word_embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size_w], -1.0, 1.0))

            # embedding for graphs
            self.doc_embeddings = tf.Variable(
                tf.random_uniform([self.document_size, self.embedding_size_d], -1.0, 1.0))

            if self.concat: # concatenating word vectors and doc vector
                combined_embed_vector_length = self.embedding_size_w * self.window_size + self.embedding_size_d
            else: # concatenating the average of word vectors and the doc vector
                combined_embed_vector_length = self.embedding_size_w + self.embedding_size_d

            # softmax weights, W and D vectors should be concatenated before applying softmax
            self.weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, combined_embed_vector_length],
                    stddev=1.0 / math.sqrt(combined_embed_vector_length)))
            # softmax biases
            self.biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # shape: (batch_size, embeddings_size)
            embed = [] # collect embedding matrices with shape=(batch_size, embedding_size)
            if self.concat:
                for j in range(self.window_size):
                    embed_w = tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
                    embed.append(embed_w)
            else:
                # averaging word vectors
                embed_w = tf.zeros([self.batch_size, self.embedding_size_w])
                for j in range(self.window_size):
                    embed_w += tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
                embed.append(embed_w)

            embed_d = tf.nn.embedding_lookup(self.doc_embeddings, self.train_dataset[:, self.window_size])
            embed.append(embed_d)
            # concat word and doc vectors
            self.embed = tf.concat(embed, 1)

            # choosing negative sampling function
            sampled_values = None # log uniform by default
            if self.candidate_func == 'uniform': # change to uniform
                sampled_values = tf.nn.uniform_candidate_sampler(
                    true_classes=tf.to_int64(self.train_labels),
                    num_true=1,
                    num_sampled=self.num_samples,
                    unique=True,
                    range_max=self.vocabulary_size)

            # Compute the loss, using a sample of the negative labels each time.
            loss = None
            if self.loss_type == 'sampled_softmax':
                loss = tf.nn.sampled_softmax_loss(self.weights, self.biases, self.train_labels,
                                                  self.embed,
                                                  self.num_samples,
                                                  self.vocabulary_size,
                                                  sampled_values = sampled_values)
            elif self.loss_type == 'nce':
                loss = tf.nn.nce_loss(self.weights, self.biases, self.train_labels,
                                     self.embed, self.num_samples, self.vocabulary_size,
                                     sampled_values=sampled_values)

            self.loss = tf.reduce_mean(loss)

            # Optimizer.
            if self.optimize == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
            elif self.optimize == 'SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            # Normalize embeddings
            norm_w = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True))
            self.normalized_word_embeddings = self.word_embeddings/norm_w

            norm_d = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = self.doc_embeddings/norm_d

            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def _train_thread_body(self):
        '''Train model on random anonymous walk batches.'''
        label_suffix = ''
        if self.graph_labels is not None:
            label_suffix = '_' + graph_labels

        while True:
            batch_data, batch_labels = self.g2v.generate_file_batch(batch_size, window_size, self.doc_id,
                                                                    self.ROOT + self.dataset + '_corpus{}/{}'.format(label_suffix, self.corpus_fn_name.format(self.doc_id)),
                                                                    self.nodes_per_graphs[self.doc_id])
            # batch_data, batch_labels = self.g2v.generate_random_batch(batch_size=self.batch_size,
            #                                                         window_size=self.window_size,
            #                                                         steps=self.steps, walk_ids=self.walk_ids,
            #                                                         doc_id=self.doc_id,
            #                                                         graph_labels = self.graph_labels)
            feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
            op, l = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            self.sample += 1
            self.global_step += 1

            self.average_loss += l
            # The average loss is an estimate of the loss over the last 100 batches.
            # if self.global_step % 100 == 0:
                # print('Average loss at step %d: %f' % (self.global_step, self.average_loss))
                # self.average_loss = 0

            if self.sample >= self.batches_per_epoch:
                break

    def train(self):
        '''Train the model.'''
        session = self.sess

        session.run(self.init_op)

        self.average_loss = 0
        self.global_step = 0
        print('Initialized')
        random_order = list(range(len(self.sorted_graphs)))
        random.shuffle(random_order)
        for ep in range(self.epochs):
            print('Epoch: {}'.format(ep))
            time2epoch = time.time()
            for rank_id, doc_id in enumerate(random_order):
            # for doc_id, graph_fn in enumerate(self.sorted_graphs):
            #     graph_fn = self.sorted_graphs[doc_id]

                time2graph = time.time()
                self.sample = 0
                self.doc_id = doc_id
                # self.g2v.read_graphml(self.folder + graph_fn)
                # self.g2v.create_random_walk_graph()

                # print('{}-{}. Graph-{}: {} nodes'.format(ep, rank_id, doc_id, len(self.g2v.rw_graph)))
                # if self.flag2iterations == True: # take sample of N words per each graph with N nodes
                #     self.batches_per_epoch = len(self.g2v.rw_graph)

                self._train_thread_body()

                if rank_id > 0 and not rank_id%100:
                    print('Graph {}-{}: {:.2f}'.format(ep, rank_id, time.time() - time2graph))
            print('Time for epoch {}: {:.2f}'.format(ep, time.time() - time2epoch))
            # save temporary embeddings
            if not ep%10:
                self.graph_embeddings = session.run(self.normalized_doc_embeddings)
                np.savez_compressed(RESULTS_FOLDER + '/' + dataset +  '/tmp/embeddings_{}.txt'.format(ep), E=self.graph_embeddings)

        self.graph_embeddings = session.run(self.normalized_doc_embeddings)

        return self