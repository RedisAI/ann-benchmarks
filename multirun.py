from os import system, path, makedirs
from multiprocessing import Process
import argparse
import time
import json
from redis import Redis
from redis.cluster import RedisCluster
import h5py
import os
import pathlib
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
        '--json-output',
        help='Path to the output file. If defined will store the results in json format.',
        default=""
    )
    parser.add_argument(
        "-k", "--count",
        default="10",
        type=str,
        help="the number of near neighbours to search for")
    parser.add_argument(
        '--host',
        help='host name or IP',
        default=None)
    parser.add_argument(
        '--port',
        type=str,
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
        type=str,
        metavar='NUM',
        help='total number of clients running in parallel to build the index (could be 0)',
        default="1")
    parser.add_argument(
        '--test-clients',
        type=str,
        metavar='NUM',
        help='total number of clients running in parallel to test the index (could be 0)',
        default="1")
    parser.add_argument(
        '--force',
        help='re-run algorithms even if their results already exist',
        action='store_true')
    parser.add_argument(
        '--algorithm',
        metavar='ALGO',
        help='run redisearch with this algorithm',
        default="redisearch-hnsw")
    parser.add_argument(
        '--run-group',
        metavar='NAME',
        help='run only the named run group',
        default=None)
    parser.add_argument(
        '--runs',
        type=str,
        help='run each algorithm instance %(metavar)s times and use only'
             ' the best result',
        default="3")
    parser.add_argument(
        '--cluster',
        action='store_true',
        help='working with a cluster')

    args = parser.parse_args()
    isredis = True if 'redisearch' in args.algorithm else False

    if isredis:
        redis = RedisCluster if args.cluster else Redis
        redis = redis(host=args.host, port=int(args.port), password=args.auth, username=args.user)


    base = 'python3 run.py --local --algorithm ' + args.algorithm + ' -k ' + args.count + \
           ' --dataset ' + args.dataset + " --runs {} ".format(args.runs)

    if args.host:       base += ' --host ' + str(args.host)
    if args.port:       base += ' --port ' + str(args.port)
    if args.user:       base += ' --user ' + str(args.user)
    if args.auth:       base += ' --auth ' + str(args.auth)
    if args.force:      base += ' --force'
    if args.cluster:    base += ' --cluster'
    if args.run_group:  base += ' --run-group ' + str(args.run_group)

    base_build = base + ' --build-only --total-clients ' + str(args.build_clients)
    base_test = base + ' --test-only --runs 1 --total-clients ' + str(args.test_clients)
    workdir = pathlib.Path(__file__).parent.absolute()
    print("Changing the workdir to {}".format(workdir))
    os.chdir(workdir)
    results_dict = {}
    if int(args.build_clients) > 0:
        clients = [Process(target=system, args=(base_build + ' --client-id ' + str(i),)) for i in range(1, int(args.build_clients) + 1)]

        t0 = time.time()
        for client in clients: client.start()
        for client in clients: client.join()
        total_time = time.time() - t0
        print(f'total build time: {total_time}\n\n')

        fn = get_result_filename(args.dataset, args.count)
        print(fn)
        fn = path.join(fn, args.algorithm)
        if not path.isdir(fn):
            makedirs(fn)
        fn = path.join(fn, 'build_stats.hdf5')
        f = h5py.File(fn, 'w')
        f.attrs["build_time"] = total_time
        print(fn)
        index_size = -1
        if isredis:
            if args.cluster:
                index_size = -1 # TODO: get total size from all the shards
            else:
                index_size = redis.ft('ann_benchmark').info()['vector_index_sz_mb']
            f.attrs["index_size"] = index_size
        f.close()
        results_dict["build"] = {"total_clients":args.build_clients, "build_time": total_time, "vector_index_sz_mb": index_size }



    if int(args.test_clients) > 0:
        queriers = [Process(target=system, args=(base_test + ' --client-id ' + str(i),)) for i in range(1, args.test_clients + 1)]
        t0 = time.time()
        for querier in queriers: querier.start()
        for querier in queriers: querier.join()
        query_time = time.time() - t0
        print(f'total test time: {query_time}')
        results_dict["query"] = {"total_clients":args.test_clients, "test_time": query_time }

    if args.json_output != "":
        with open(args.json_output,"w")as json_out_file:
            print(f'storing json result into: {args.json_output}')
            json.dump(results_dict,json_out_file)
