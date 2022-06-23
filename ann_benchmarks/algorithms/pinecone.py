from __future__ import absolute_import
from sqlite3 import paramstyle
from ann_benchmarks.algorithms.base import BaseANN
import sys
import pinecone

class Pinecone(BaseANN):
    def __init__(self, metric, dim, conn_params, type):
        pinecone.init(api_key=conn_params['auth'])
        m = {'angular': 'cosine', 'euclidean': 'euclidean'}[metric]
        self.name = 'ann-benchmark'
        if self.name not in pinecone.list_indexes():
            pinecone.create_index(self.name, dimension=dim, metric=m,
                                  index_type=type, shards=int(conn_params["shards"]), )
        self.index = pinecone.Index(self.name)

    def fit(self, X, offset=0, limit=None):
        limit = limit if limit else len(X)

        bulk = [(str(i), X[i].tolist()) for i in range(offset, limit)]
        # approximation for pinecone insert limit (2MB or 1000 vectors)
        batch_size = min(1000, 2 * 1024 * 1024 // (sys.getsizeof(bulk[-1]))) # bulk[-1] should be the largest (longest name)

        for batch in [bulk[i: i+batch_size] for i in range(0, len(bulk), batch_size)]:
            # print(f'inserting vectors {batch[0][0]} to {batch[-1][0]}')
            self.index.upsert(batch)
        
        # print(self.index.describe_index_stats())
        # print(pinecone.describe_index(self.name))
    
    def query(self, v, n):
        res = self.index.query(v.tolist(), top_k=n)
        return [int(e['id']) for e in res['matches']]
    
    def freeIndex(self):
        pinecone.delete_index(self.name)

    def __str__(self):
        return f'Pinecone({pinecone.describe_index(self.name)})'
