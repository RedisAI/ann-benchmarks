"""
ann-benchmarks interfaces for Elasticsearch.
Note that this requires X-Pack, which is not included in the OSS version of Elasticsearch.
"""
import logging
from time import sleep
from os import environ
from urllib.error import URLError

from elasticsearch import Elasticsearch, BadRequestError
from elasticsearch.helpers import bulk
from elastic_transport.client_utils import DEFAULT

from ann_benchmarks.algorithms.base import BaseANN

# Configure the elasticsearch logger.
# By default, it writes an INFO statement for every request.
logging.getLogger("elasticsearch").setLevel(logging.WARN)

# Uncomment these lines if you want to see timing for every HTTP request and its duration.
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("elasticsearch").setLevel(logging.INFO)

def es_wait(es):
    print("Waiting for elasticsearch health endpoint...")
    for i in range(30):
        try:
            res = es.cluster.health(wait_for_status='yellow', timeout='1s')
            if not res['timed_out']: # then status is OK
                print("Elasticsearch is ready")
                return
        except URLError:
            pass
        sleep(1)
    raise RuntimeError("Failed to connect to elasticsearch server")


class ElasticsearchScriptScoreQuery(BaseANN):
    """
    KNN using the Elasticsearch dense_vector datatype and script score functions.
    - Dense vector field type: https://www.elastic.co/guide/en/elasticsearch/reference/master/dense-vector.html
    - Dense vector queries: https://www.elastic.co/guide/en/elasticsearch/reference/master/query-dsl-script-score-query.html
    """

    def __init__(self, metric: str, dimension: int, conn_params, method_param):
        self.name = f"elasticsearch-script-score-query_metric={metric}_dimension={dimension}_params{method_param}"
        self.metric = {"euclidean": 'l2_norm', "angular": 'cosine'}[metric]
        self.method_param = method_param
        self.dimension = dimension
        self.timeout = 60 * 60
        h = conn_params['host'] if conn_params['host'] is not None else 'localhost'
        p = conn_params['port'] if conn_params['port'] is not None else '9200'
        u = conn_params['user'] if conn_params['user'] is not None else 'elastic'
        a = conn_params['auth'] if conn_params['auth'] is not None else ''
        self.index = "ann_benchmark"
        self.shards = conn_params['shards']
        try:
            self.es = Elasticsearch(f"http://{h}:{p}",  request_timeout=self.timeout, basic_auth=(u, a), refresh_interval=-1)
            self.es.info()
        except Exception:
            self.es = Elasticsearch(f"https://{h}:{p}", request_timeout=self.timeout, basic_auth=(u, a), ca_certs=environ.get('ELASTIC_CA', DEFAULT))
        self.batch_res = []
        es_wait(self.es)

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
        try:
            self.es.indices.create(index=self.index, mappings=mappings, settings=dict(number_of_shards=self.shards, number_of_replicas=0))
        except BadRequestError as e:
            if 'resource_already_exists_exception' not in e.message: raise e

        def gen():
            for i, vec in enumerate(X):
                yield { "_op_type": "index", "_index": self.index, "vec": vec.tolist(), 'id': str(i) }

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
        return [int(h['fields']['id'][0]) for h in res['hits']['hits']]

    def batch_query(self, X, n):
        self.batch_res = [self.query(q, n) for q in X]

    def get_batch_results(self):
        return self.batch_res

