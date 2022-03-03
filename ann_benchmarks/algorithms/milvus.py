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
    def __init__(self, metric, dim, conn_params, index_type, method_params):
        self._host = conn_params['host']
        self._port = conn_params['port'] # 19530
        self._index_type = index_type
        self._method_params = method_params
        self._metric = {'angular': 'IP', 'euclidean': 'L2'}[metric]
        self._query_params = dict()
        connections.connect(host=conn_params['host'], port=conn_params['port'])
        try:
            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
            ]
            schema = CollectionSchema(fields)
            self._milvus = Collection('milvus', schema)
            self._milvus.create_index('vector', {'index_type': self._index_type, 'metric_type':self._metric, 'params':self._method_params})
        except:
            self._milvus = Collection('milvus')
    
    def fit(self, X, offset=0, limit=None):
        limit = limit if limit else len(X)
        X = X[offset:limit]
        if self._metric == 'IP':
            X = sklearn.preprocessing.normalize(X)

        self._milvus.insert([[id for id in range(offset, limit)], X.tolist()])

    def set_query_arguments(self, param):
        if self._milvus.has_index():
            if utility.wait_for_index_building_complete('milvus', 'vector'):
                self._milvus.load()
            else: raise Exception('index has error')
        else: raise Exception('index is missing')
        if 'IVF_' in self._index_type:
            if param > self._method_params['nlist']:
                print('warning! nprobe > nlist')
                param = self._method_params['nlist']
            self._query_params['nprobe'] = param
        if 'HNSW' in self._index_type:
            self._query_params['ef'] = param

    def query(self, v, n):
        if self._metric == 'IP':
            v /= numpy.linalg.norm(v)
        v = v.tolist()
        results = self._milvus.search([v], 'vector', {'metric_type':self._metric, 'params':self._query_params}, limit=n)
        if not results:
            return []  # Seems to happen occasionally, not sure why
        result_ids = [result.id for result in results[0]]
        return result_ids

    def __str__(self):
        return 'Milvus(index_type=%s, method_params=%s, query_params=%s)' % (self._index_type, str(self._method_params), str(self._query_params))

    def freeIndex(self):
        utility.drop_collection("mlivus")
