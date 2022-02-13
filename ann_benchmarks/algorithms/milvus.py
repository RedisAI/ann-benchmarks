from __future__ import absolute_import
from sqlite3 import paramstyle
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    IndexType,
    Collection,
)
import numpy
import sklearn.preprocessing
from ann_benchmarks.algorithms.base import BaseANN


class Milvus(BaseANN):
    def __init__(self, metric, conn_params, index_type, method_params):
        self._host = conn_params['host']
        self._port = conn_params['port'] # 19530
        # connections.connect(host=conn_params['host'], port=conn_params['port'])
        # fields = [
        #     FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        #     FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=100)
        # ]
        # schema = CollectionSchema(fields)
        # self._milvus = Collection('milvus', schema)
        self._index_type = index_type
        self._method_params = method_params
        self._nprobe = None
        self._metric = metric

    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1)
        
        # TODO: if we can set the dim later, mabe return this to the init func
        connections.connect(host=self._host, port=self._port)
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=len(X[0]))
        ]
        schema = CollectionSchema(fields)
        self._milvus = Collection('milvus', schema)

        self._milvus.insert([[id for id in range(len(X))], X.tolist()])
        self._milvus.create_index('vector', {'index_type': self._index_type, 'metric_type':'L2', 'params':self._method_params})
        self._milvus.load()

    def set_query_arguments(self, param):
        self._query_params = dict()
        if 'IVF_' in self._index_type:
            if param > self._method_params['nlist']:
                print('warning! nprobe > nlist')
                param = self._method_params['nlist']
            self._query_params['nprobe'] = param
        if 'HNSW' in self._index_type:
            self._query_params['ef'] = param

    def query(self, v, n):
        if self._metric == 'angular':
            v /= numpy.linalg.norm(v)
        v = v.tolist()
        results = self._milvus.search([v], 'vector', {'metric_type':'L2', 'params':self._query_params}, limit=n)
        if not results:
            return []  # Seems to happen occasionally, not sure why
        result_ids = [result.id for result in results[0]]
        return result_ids

    def __str__(self):
        return 'Milvus(index_type=%s, method_params=%s, query_params=%s)' % (self._index_type, str(self._method_params), str(self._nprobe))
