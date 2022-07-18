"""
ann-benchmarks interfaces for Elasticsearch.
Note that this requires X-Pack, which is not included in the OSS version of Elasticsearch.
"""
import logging
from time import sleep
from os import environ
from urllib.error import URLError
from urllib.request import Request, urlopen

from elasticsearch import Elasticsearch, ConnectionTimeout, BadRequestError
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
            if not res['timed_out']:  # then status is OK
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
        self.es = Elasticsearch(f"{h}:{p}", request_timeout=self.timeout, basic_auth=(u, a),
                                ca_certs=environ.get('ELASTIC_CA', DEFAULT), timeout=180, max_retries=10, retry_on_timeout=True)
        self.batch_res = []
        es_wait(self.es)
        self.check_index_does_not_exist()
        self.create_index()

    def wait_for_readiness(self):
        ready = False
        for i in range(self.timeout):
            stats = self.es.indices.stats(index=self.index)
            if stats['_shards']['total'] == stats['_shards']['successful']:
                ready = True
                break
            sleep(1)
        return ready

    """
    Based uppon https://www.elastic.co/guide/en/elasticsearch/reference/current/tune-for-indexing-speed.html
    and given we're doing bulk index operations and only after fully indexing we do the querying we should tune
    the index refresh interval to larger timeframes ( default is 1s ) or even disable it while ingesting.
    """

    def update_refresh_interval(self, new_refresh_interval):
        previous_settings = self.es.indices.get_settings(index=self.index)
        current_refresh_interval = "default one"
        try:
            current_refresh_interval = previous_settings[self.index]['settings']['index']['refresh_interval']
        except KeyError:
            pass
        print("Altering index refresh interval from {} to {}".format(
            current_refresh_interval, new_refresh_interval ))
        self.es.indices.put_settings(index=self.index, settings={
            "index.refresh_interval": new_refresh_interval
        })

    def fit(self, X):
        def gen():
            for i, vec in enumerate(X):
                yield {"_op_type": "index", "_index": self.index, "vec": vec.tolist(), 'id': str(i)}

        (_, errors) = bulk(self.es, gen(), chunk_size=500, max_retries=10)
        assert len(errors) == 0, errors

    def create_index(self):
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
        print("Creating elastic index named {}".format(self.index))
        print("\t\tIndex properties {}".format(mappings))
        try:
            self.es.indices.create(index=self.index, mappings=mappings,
                                   settings=dict(number_of_shards=1, number_of_replicas=0), timeout=f'{self.timeout}m')
        except ConnectionTimeout as e:
            if not self.wait_for_readiness():
                raise e
        except BadRequestError as e:
            if 'resource_already_exists_exception' not in e.message:
                raise e
        print("Setting refresh interval to 60 seconds")
        self.update_refresh_interval(60)

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, q, n):
        knn = dict(field='vec', query_vector=q.tolist(),
                   k=n, num_candidates=self.ef)
        res = self.es.knn_search(index=self.index, knn=knn, source=False, docvalue_fields=['id'],
                                 stored_fields="_none_", filter_path=["hits.hits.fields.id"])
        return [int(h['fields']['id'][0]) for h in res['hits']['hits']]

    def batch_query(self, X, n):
        self.batch_res = [self.query(q, n) for q in X]

    def get_batch_results(self):
        return self.batch_res

    def freeIndex(self):
        print("Deleting elastic index named {}".format(self.index))
        self.es.indices.delete(index=self.index)

    def check_index_does_not_exist(self):
        print("Checking if index named {} exists.".format(self.index))
        res = self.es.indices.get_alias("*")
        print("Indices: {}".format(res))
        if self.index in res:
            print("Detected index. deleting it...")
            self.freeIndex()
        else:
            print("No index detected.")
