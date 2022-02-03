from os import system, path, makedirs
from multiprocessing import Process
import argparse
import time
from redis import Redis
import h5py
from ann_benchmarks.main import positive_int
from ann_benchmarks.results import get_result_filename
# from ann_benchmarks.datasets import DATASETS

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
        default=None)
    parser.add_argument(
        '--port',
        type=positive_int,
        help='the port "host" is listening on',
        default=None)
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
        type=positive_int,
        help='the port "host" is listening on',
        default=1)
    parser.add_argument(
        '--test-clients',
        type=positive_int,
        help='the port "host" is listening on',
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

    args = parser.parse_args()

    base = 'python run.py --local --algorithm redisearch-' + args.algorithm.lower() + ' --total-clients ' + str(args.build_clients) +\
           ' -k ' + str(args.count) + ' --dataset ' + args.dataset

    if args.user:   base += ' --user ' + str(args.user)
    if args.auth:   base += ' --auth ' + str(args.auth)
    if args.host:   base += ' --host ' + str(args.host)
    if args.port:   base += ' --port ' + str(args.port)
    if args.force:  base += ' --force'

    base_build = base + ' --build-only --total-clients ' + str(args.build_clients)
    base_test = base + ' --test-only --runs 1 --total-clients ' + str(args.test_clients)

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
    f.attrs["index_size"] = 1000 # demo - add client to get data usage
    f.close()

    queriers = [Process(target=system, args=(base_test + ' --client-id ' + str(i),)) for i in range(1, args.test_clients + 1)]
    t0 = time.time()
    for querier in queriers: querier.start()
    for querier in queriers: querier.join()
    query_time = time.time() - t0
    print(f'total test time: {query_time}')
