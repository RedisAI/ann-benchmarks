from multiprocessing import Process
import argparse
import time
import json
from numpy import average
from redis import Redis
from redis.cluster import RedisCluster
import h5py
import os
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import pathlib
from ann_benchmarks.results import get_result_filename

def aggregate_outputs(files, clients):
    different_attrs = set([f.split('client')[0] for f in files])
    groups = [[f + f'client_{i}.hdf5' for i in range(1, clients + 1)] for f in different_attrs]

    if len(different_attrs) * clients > len(files):
        print(f'missing files! got {len(files)} but expected {len(different_attrs) * clients}')
        print('got files:')
        [print('\t'+f) for f in files]
        print('probably missing files:')
        [[print('\t'+f) for f in g if f not in files] for g in groups]
        assert False
    elif len(different_attrs) * clients < len(files):
        print(f'too many files! got {len(files)} but expected {len(different_attrs) * clients}')
        print('got files:')
        [print('\t'+f) for f in files]
        print('probably unnecessary files:')
        [print('\t'+f) for f in files if len([g for g in groups if f in g]) == 0]
        raise False
        
    for group in groups:
        fn = group[0].split('client')[0][:-1] + '.hdf5'
        f = h5py.File(fn, 'w')
        
        fs = [h5py.File(fi, 'r') for fi in group]
        for k, v in fs[0].attrs.items():
            f.attrs[k] = v
        f.attrs["best_search_time"] = average([fi.attrs["best_search_time"] for fi in fs])
        f.attrs["candidates"] = average([fi.attrs["candidates"] for fi in fs])
        
        times = f.create_dataset('times', fs[0]['times'].shape, 'f')
        neighbors = f.create_dataset('neighbors', fs[0]['neighbors'].shape, 'i')
        distances = f.create_dataset('distances', fs[0]['distances'].shape, 'f')
        num_tests = len(times)
        
        for i in range(num_tests):
            neighbors[i] = [n for n in fs[0]['neighbors'][i]]
            distances[i] = [n for n in fs[0]['distances'][i]]
            times[i] = average([fi['times'][i] for fi in fs])
        
        [fi.close() for fi in fs]
        [os.remove(fi) for fi in group]
        f.close()

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
        type=str,
        help='host name or IP',
        default=None)
    parser.add_argument(
        '--port',
        type=str,
        help='the port "host" is listening on',
        default=None)
    parser.add_argument(
        '--auth', '-a',
        type=str,
        metavar='PASS',
        help='password for connection',
        default=None)
    parser.add_argument(
        '--user',
        type=str,
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
        type=str,
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

    if args.host is None:
        args.host = 'localhost'
    if args.port is None:
        if 'redisearch' in args.algorithm: args.port = '6379'
        if 'milvus' in args.algorithm: args.port = '19530'

    if isredis:
        redis = RedisCluster if args.cluster else Redis
        redis = redis(host=args.host, port=int(args.port), password=args.auth, username=args.user)


    base = 'python3 run.py --local --algorithm ' + args.algorithm + ' -k ' + args.count + ' --dataset ' + args.dataset

    if args.host:       base += ' --host ' + args.host
    if args.port:       base += ' --port ' + args.port
    if args.user:       base += ' --user ' + args.user
    if args.auth:       base += ' --auth ' + args.auth
    if args.force:      base += ' --force'
    if args.cluster:    base += ' --cluster'
    if args.run_group:  base += ' --run-group ' + args.run_group

    base_build = base + ' --build-only --total-clients ' + args.build_clients
    base_test = base + ' --test-only --runs {} --total-clients {}'.format(args.runs, args.test_clients)
    workdir = pathlib.Path(__file__).parent.absolute()
    print("Changing the workdir to {}".format(workdir))
    os.chdir(workdir)
    outputsdir = "{}/{}".format(workdir, get_result_filename(args.dataset, args.count))
    outputsdir = os.path.join(outputsdir, args.algorithm)
    if not os.path.isdir(outputsdir):
        os.makedirs(outputsdir)
    results_dict = {}
    if int(args.build_clients) > 0:
        clients = [Process(target=os.system, args=(base_build + ' --client-id ' + str(i),)) for i in range(1, int(args.build_clients) + 1)]

        t0 = time.time()
        for client in clients: client.start()
        for client in clients: client.join()
        total_time = time.time() - t0
        print(f'total build time: {total_time}\n\n')

        fn = os.path.join(outputsdir, 'build_stats')
        f = h5py.File(fn, 'w')
        f.attrs["build_time"] = total_time
        print(fn)
        index_size = -1
        if isredis:
            if not args.cluster: # TODO: get total size from all the shards
                index_size = redis.ft('ann_benchmark').info()['vector_index_sz_mb']
            f.attrs["index_size"] = float(index_size)
        f.close()
        results_dict["build"] = {"total_clients":args.build_clients, "build_time": total_time, "vector_index_sz_mb": index_size }

    if int(args.test_clients) > 0:
        queriers = [Process(target=os.system, args=(base_test + ' --client-id ' + str(i),)) for i in range(1, int(args.test_clients) + 1)]
        test_stats = set()
        watcher = PatternMatchingEventHandler(["*.hdf5"], ignore_directories=True )
        def on_created_or_modified(event):
            test_stats.add(event.src_path)
        watcher.on_created = on_created_or_modified
        watcher.on_modified = on_created_or_modified
        observer = Observer()
        observer.schedule(watcher, workdir, True)
        observer.start()
        t0 = time.time()
        for querier in queriers: querier.start()
        for querier in queriers: querier.join()
        query_time = time.time() - t0
        print(f'total test time: {query_time}')
        observer.stop()
        observer.join()
        results_dict["query"] = {"total_clients":args.test_clients, "test_time": query_time }
        print(f'summarizing {int(args.test_clients)} clients data ({len(test_stats)} files into {len(test_stats) // int(args.test_clients)})...')
        aggregate_outputs(test_stats, int(args.test_clients))
        print('done!')

    if args.json_output != "":
        with open(args.json_output,"w")as json_out_file:
            print(f'storing json result into: {args.json_output}')
            json.dump(results_dict,json_out_file)
