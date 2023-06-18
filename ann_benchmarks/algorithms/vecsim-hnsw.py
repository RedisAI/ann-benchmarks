from __future__ import absolute_import
import os
from VecSim import *
import numpy as np
from ann_benchmarks.constants import INDEX_DIR
from ann_benchmarks.algorithms.base import BaseANN


class VecSimHnsw(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {'angular': VecSimMetric_Cosine, 'euclidean': VecSimMetric_L2}[metric]
        self.method_param = method_param
        # print(self.method_param,save_index,query_param)
        self.ef = None
        self.name = 'VecSim-hnsw (%s)' % (self.method_param)

    def fit(self, X):
        hnswparams = HNSWParams()
        hnswparams.M =self.method_param['M']
        hnswparams.efConstruction = self.method_param['efConstruction']
        hnswparams.initialCapacity = len(X)
        hnswparams.dim = len(X[0])
        hnswparams.type = VecSimType_FLOAT32
        hnswparams.metric = self.metric
        hnswparams.multi = False

        self.index = HNSWIndex(hnswparams)

        for i, vector in enumerate(X):
            self.index.add_vector(vector, i)

    def set_query_arguments(self, ef):
        self.ef = ef
        self.index.set_ef(ef)

    def query(self, v, n):
        return self.index.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]

    def freeIndex(self):
        del self.index

    def __str__(self):
        return f"{self.name}, efRuntime: {self.ef}"
