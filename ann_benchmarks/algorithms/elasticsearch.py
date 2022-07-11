"""
ann-benchmarks interfaces for Elasticsearch.
Note that this requires X-Pack, which is not included in the OSS version of Elasticsearch.
"""
import logging
from time import sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from ann_benchmarks.algorithms.base import BaseANN

# Configure the elasticsearch logger.
# By default, it writes an INFO statement for every request.
logging.getLogger("elasticsearch").setLevel(logging.WARN)

# Uncomment these lines if you want to see timing for every HTTP request and its duration.
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("elasticsearch").setLevel(logging.INFO)

def es_wait():
    print("Waiting for elasticsearch health endpoint...")
    req = Request("http://localhost:9200/_cluster/health?wait_for_status=yellow&timeout=1s")
    for i in range(30):
        try:
            res = urlopen(req)
            if res.getcode() == 200:
                print("Elasticsearch is ready")
                return
        except URLError:
            pass
        sleep(1)
    raise RuntimeError("Failed to connect to local elasticsearch")


class ElasticsearchScriptScoreQuery(BaseANN):
    """
    KNN using the Elasticsearch dense_vector datatype and script score functions.
    - Dense vector field type: https://www.elastic.co/guide/en/elasticsearch/reference/master/dense-vector.html
    - Dense vector queries: https://www.elastic.co/guide/en/elasticsearch/reference/master/query-dsl-script-score-query.html
    """

    def __init__(self, metric: str, dimension: int, method_param):
        self.name = f"elasticsearch-script-score-query_metric={metric}_dimension={dimension}_params{method_param}"
        self.metric = {"euclidean": 'l2_norm', "angular": 'cosine'}[metric]
        self.method_param = method_param
        self.dimension = dimension
        self.index = f"es-ssq-{metric}-{dimension}"
        self.es = Elasticsearch(["http://localhost:9200"])
        self.batch_res = []
        es_wait()

    def fit(self, X):
        mappings = dict(
            properties=dict(
                id=dict(type="keyword", store=True),
                vec=dict(
                    type="dense_vector",
                    dims=self.dimension,
                    similarity=self.metric,
                    index=True,
                    index_options=self.method_param
                )
            )
        )
        if self.es.indices.exists(index=self.index):
            print('deleteing...', end=' ')
            self.es.indices.delete(index=self.index)
            print('done!')
        self.es.indices.create(index=self.index, mappings=mappings, settings=dict(number_of_shards=1, number_of_replicas=0))

        # self.es.indices.put_mapping(properties=properties, index=self.index)

        def gen():
            for i, vec in enumerate(X):
                yield { "_op_type": "index", "_index": self.index, "vec": vec.tolist(), 'id': str(i + 1) }

        (_, errors) = bulk(self.es, gen(), chunk_size=500, max_retries=9)
        assert len(errors) == 0, errors

        self.es.indices.refresh(index=self.index)
        self.es.indices.forcemerge(index=self.index, max_num_segments=1)

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, q, n):
        knn = dict(field='vec', query_vector=q.tolist(), k=n, num_candidates=self.ef)
        res = self.es.knn_search(index=self.index, knn=knn, source=False, docvalue_fields=['id'],
                                 stored_fields="_none_", filter_path=["hits.hits.fields.id"])
        return [int(h['fields']['id'][0]) - 1 for h in res['hits']['hits']]

    def batch_query(self, X, n):
        self.batch_res = [self.query(q, n) for q in X]

    def get_batch_results(self):
        return self.batch_res

