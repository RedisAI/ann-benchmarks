from os import system, path, makedirs
from multiprocessing import Process
import argparse
import time
from redis import Redis
from redis.cluster import RedisCluster
import h5py
from ann_benchmarks.main import positive_int
from ann_benchmarks.results import get_result_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset',
        metavar='NAME',
        help='the dataset to load training points from',
        default='glove-100-angular')
    parser.add_argument(
        "-k", "--count",
        default=10,
        type=positive_int,
        help="the number of near neighbours to search for")
    parser.add_argument(
        '--host',
        help='host name or IP',
        default='localhost')
    parser.add_argument(
        '--port',
        type=positive_int,
        help='the port "host" is listening on',
        default=6379)
    parser.add_argument(
        '--auth', '-a',
        metavar='PASS',
        help='password for connection',
        default=None)
    parser.add_argument(
        '--user',
        metavar='NAME',
        help='user name for connection',
        default=None)
    parser.add_argument(
        '--build-clients',
        type=int,
        metavar='NUM',
        help='total number of clients running in parallel to build the index (could be 0)',
        default=1)
    parser.add_argument(
        '--test-clients',
        type=int,
        metavar='NUM',
        help='total number of clients running in parallel to test the index (could be 0)',
        default=1)
    parser.add_argument(
        '--force',
        help='re-run algorithms even if their results already exist',
        action='store_true')
    parser.add_argument(
        '--algorithm',
        metavar='ALGO',
        help='run redisearch with this algorithm',
        default="hnsw")
    parser.add_argument(
        '--cluster',
        action='store_true',
        help='working with a cluster')

    args = parser.parse_args()

    redis = RedisCluster if args.cluster else Redis
    redis = redis(host=args.host, port=args.port, password=args.auth, username=args.user)

    base = 'python run.py --local --algorithm redisearch-' + args.algorithm.lower() + ' -k ' + str(args.count) + \
           ' --dataset ' + args.dataset + ' --host ' + str(args.host) + ' --port ' + str(args.port)

    if args.user:   base += ' --user ' + str(args.user)
    if args.auth:   base += ' --auth ' + str(args.auth)
    if args.force:  base += ' --force'
    if args.cluster:base += ' --cluster'

    base_build = base + ' --build-only --total-clients ' + str(args.build_clients)
    base_test = base + ' --test-only --runs 1 --total-clients ' + str(args.test_clients)

    if args.build_clients > 0:
        clients = [Process(target=system, args=(base_build + ' --client-id ' + str(i),)) for i in range(1, args.build_clients + 1)]

        t0 = time.time()
        for client in clients: client.start()
        for client in clients: client.join()
        total_time = time.time() - t0
        print(f'total build time: {total_time}\n\n')

        fn = get_result_filename(args.dataset, args.count)
        if not path.isdir(fn):
            makedirs(fn)
        fn = path.join(fn, 'build_stats.hdf5')
        f = h5py.File(fn, 'w')
        f.attrs["build_time"] = total_time
        if args.cluster:
            f.attrs["index_size"] = -1 # TODO: get total size from all the shards
        else:
            f.attrs["index_size"] = redis.ft('ann_benchmark').info()['vector_index_sz_mb']*0x100000
        f.close()

    if args.test_clients > 0:
        queriers = [Process(target=system, args=(base_test + ' --client-id ' + str(i),)) for i in range(1, args.test_clients + 1)]
        t0 = time.time()
        for querier in queriers: querier.start()
        for querier in queriers: querier.join()
        query_time = time.time() - t0
        print(f'total test time: {query_time}')
