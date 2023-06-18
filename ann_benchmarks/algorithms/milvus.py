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
import sys


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
            if utility.has_collection('milvus'):
                self._milvus = Collection('milvus')
            else:
                self._milvus = Collection('milvus', schema)
        except:
            self._milvus = Collection('milvus')
        print('initialization completed!')
    
    def fit(self, X, offset=0, limit=None):
        limit = limit if limit else len(X)
        X = X[offset:limit]
        if self._metric == 'IP':
            X = sklearn.preprocessing.normalize(X)

        X = X.tolist()
        bulk_size = 1000 * 1024 * 1024 // (sys.getsizeof(X[0])) # approximation for milvus insert limit (1024MB)
        for bulk in [X[i: i+bulk_size] for i in range(0, len(X), bulk_size)]:
            print(f'inserting vectors {offset} to {offset + len(bulk) - 1}')
            self._milvus.insert([list(range(offset, offset + len(bulk))), bulk])
            offset += len(bulk)

        if not self._milvus.has_index():
            print('indexing...', end=' ')
            try:
                self._milvus.create_index('vector', {'index_type': self._index_type, 'metric_type':self._metric, 'params':self._method_params})
                print('done!')
            except:
                print('failed!')
        

    def set_query_arguments(self, param):
        if self._milvus.has_index():
            print('waiting for index... ', end='')
            if utility.wait_for_index_building_complete('milvus', 'vector'):
                print('done!')
                self._milvus.load()
                print('waiting for data to be loaded... ', end='')
                utility.wait_for_loading_complete('milvus')
                print('done!')
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

    def done(self):
        connections.disconnect('default')
