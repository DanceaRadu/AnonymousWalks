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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.AnonymousWalkKernel import AnonymousWalks

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42

class AWE:

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
                 root='../',
                 ext='graphml',
                 steps=7,
                 epochs=1,
                 batches_per_epoch=1,
                 candidate_func=None,
                 graph_labels=None,
                 regenerate_corpus=True,
                 neighborhood_size=1,
                 results_folder='./results'):
        '''
        Initialize AWE model.
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
        self.results_folder = results_folder

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
        self.neighborhood_size = neighborhood_size
        start2gen = time.time()
        self.generate_corpus()
        print('Finished {}'.format(time.time() - start2gen))

        self.vocabulary_size = max(self.walk_ids.values()) + 1
        print('Number of words: {}'.format(self.vocabulary_size))

        # init all variables in PyTorch
        self._init_graph()

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
        if self.graph_labels is not None:
            label_suffix = '_' + self.graph_labels

        corpus_path = self.ROOT + self.dataset + '_corpus' + label_suffix
        if self.regenerate_corpus or not os.path.exists(corpus_path):
            if not os.path.exists(corpus_path):
                os.mkdir(corpus_path)
            for en, graph_fn in enumerate(self.sorted_graphs):
                if en > 0 and not en % 100:
                    print(f"Graph {en}")
                g2v = AnonymousWalks()
                g2v.read_graphml(self.folder + graph_fn)
                self.nodes_per_graphs[en] = len(g2v.graph)

                g2v.write_corpus(self.neighborhood_size, self.walk_ids, self.steps, self.graph_labels,
                                 f"{corpus_path}/{self.corpus_fn_name.format(en)}")

    def _init_graph(self):
        '''
        Init a PyTorch model containing:
        input data, variables, model, loss function, optimizer
        '''
        self.model = AWEModel(self.vocabulary_size, self.embedding_size_w, self.embedding_size_d, self.document_size, self.window_size, self.concat)
        if self.optimize == 'Adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.optimize == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.criterion = nn.NLLLoss()

    def _train_thread_body(self):
        '''Train model on random anonymous walk batches.'''
        label_suffix = ''
        if self.graph_labels is not None:
            label_suffix = '_' + self.graph_labels

        corpus_path = self.ROOT + self.dataset + '_corpus' + label_suffix
        print(self.nodes_per_graphs)
        while True:
            batch_data, batch_labels = self.g2v.generate_file_batch(self.batch_size, self.window_size, self.doc_id,
                                                                    f"{corpus_path}/{self.corpus_fn_name.format(self.doc_id)}",
                                                                    self.nodes_per_graphs[self.doc_id])
            # Convert to PyTorch tensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).squeeze()

            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()

            self.sample += 1
            self.global_step += 1

            self.average_loss += loss.item()

            if self.sample >= self.batches_per_epoch:
                break

    def train(self):
        '''Train the model.'''
        self.model.train()

        self.average_loss = 0
        self.global_step = 0
        print('Initialized')
        random_order = list(range(len(self.sorted_graphs)))
        random.shuffle(random_order)
        for ep in range(self.epochs):
            print('Epoch: {}'.format(ep))
            time2epoch = time.time()
            for rank_id, doc_id in enumerate(random_order):
                time2graph = time.time()
                self.sample = 0
                self.doc_id = doc_id

                self._train_thread_body()

                if rank_id > 0 and not rank_id % 100:
                    print('Graph {}-{}: {:.2f}'.format(ep, rank_id, time.time() - time2graph))
            print('Time for epoch {}: {:.2f}'.format(ep, time.time() - time2epoch))
            # save temporary embeddings
            if not ep % 10:
                torch.save(self.model.state_dict(), f"{self.results_folder}/{self.dataset}/tmp/embeddings_{ep}.pth")

        return self.model.state_dict()

class AWEModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_size_w, embedding_size_d, document_size, window_size, concat):
        super(AWEModel, self).__init__()
        self.concat = concat
        self.window_size = window_size

        self.word_embeddings = nn.Embedding(vocabulary_size, embedding_size_w)
        self.doc_embeddings = nn.Embedding(document_size, embedding_size_d)

        if concat:
            combined_embed_vector_length = embedding_size_w * window_size + embedding_size_d
        else:
            combined_embed_vector_length = embedding_size_w + embedding_size_d

        self.linear = nn.Linear(combined_embed_vector_length, vocabulary_size)

    def forward(self, inputs):
        embed = []
        if self.concat:
            for j in range(self.window_size):
                embed_w = self.word_embeddings(inputs[:, j])
                embed.append(embed_w)
        else:
            embed_w = torch.zeros(inputs.size(0), self.word_embeddings.embedding_dim).to(inputs.device)
            for j in range(self.window_size):
                embed_w += self.word_embeddings(inputs[:, j])
            embed.append(embed_w)

        embed_d = self.doc_embeddings(inputs[:, self.window_size])
        embed.append(embed_d)

        embed = torch.cat(embed, 1)
        output = self.linear(embed)
        return output
    
dataset = 'mutag'

batch_size = 100
window_size = 8
neighborhood_size = 10
embedding_size_w = 128
embedding_size_d = 128
num_samples = 32

concat = False
loss_type = 'sampled_softmax'
optimize = 'Adagrad'
learning_rate = 1.0
root = 'Datasets/'
ext = 'graphml'
steps = 7
epochs = 1
batches_per_epoch = 50
candidate_func = None
graph_labels = None

awe = AWE(dataset = dataset, batch_size = batch_size, window_size = window_size,
                  embedding_size_w = embedding_size_w, embedding_size_d = embedding_size_d,
                  num_samples = num_samples, concat = concat, loss_type = loss_type,
                  optimize = optimize, learning_rate = learning_rate, root = root,
                  ext = ext, steps = steps, epochs = epochs, batches_per_epoch = batches_per_epoch,
                  candidate_func = candidate_func, graph_labels=graph_labels, neighborhood_size=neighborhood_size)

print()
start2emb = time.time()
print(awe.nodes_per_graphs)
awe.train() # get embeddings
finish2emb = time.time()
print()
print('Time to compute embeddings: {:.2f} sec'.format(finish2emb - start2emb))