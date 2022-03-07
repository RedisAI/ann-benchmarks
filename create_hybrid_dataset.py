from operator import ne
import click
from ann_benchmarks.datasets import get_dataset, DATASETS
from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS
import struct
import numpy as np
import click
import h5py
from joblib import Parallel, delayed
import multiprocessing

def calc_i(i, x, bf, test, neighbors, distances, count, orig_ids):
    if i % 1000 == 0:
        print('%d/%d...' % (i, len(test)))
    res = list(bf.query_with_distances(x, count))
    res.sort(key=lambda t: t[-1])
    neighbors[i] = [orig_ids[j] for j, _ in res]
    distances[i] = [d for _, d in res]
    

@click.command()
@click.option('--data_set', type=click.Choice(DATASETS.keys(), case_sensitive=False), default='glove-100-angular')
def create_ds(data_set):
    ds, dimension= get_dataset(data_set)
    bucket_0_5 = []
    bucket_1 = []
    bucket_2 = []
    bucket_5 = []
    bucket_10 = []
    bucket_20 = []
    buckets = [bucket_0_5, bucket_1, bucket_2, bucket_5, bucket_10, bucket_20]
    bucket_names=['0.5%', '1%', '2%', '5%', '10%', '20%']
    train = ds['train']
    test = ds['test']
    distance = ds.attrs['distance']
    count=len(ds['neighbors'][0])
    print(count)
    print(train.shape)
    for i in range(train.shape[0]):
        if i % 200 == 6:      # 0.5%
            bucket_0_5.append(i)
        elif i % 100 == 4:    # 1%
            bucket_1.append(i)
        elif i % 50 == 3:     # 2%
            bucket_2.append(i)
        elif i % 20 == 2:     # 5%
            bucket_5.append(i)
        elif i % 10 == 1:     # 10%
            bucket_10.append(i)
        elif i % 5 == 0:      # 20%
            bucket_20.append(i)
    print(len(bucket_0_5), len(bucket_1), len(bucket_2), len(bucket_5), len(bucket_10), len(bucket_20))
    for i, bucket in enumerate(buckets):
        fn=f'{data_set}-hybrid_{bucket_names[i]}.hd5f'
        with h5py.File(fn, 'w') as f:
            f.attrs['type'] = 'dense'
            f.attrs['distance'] = ds.attrs['distance']
            f.attrs['dimension'] = len(test[0])
            f.attrs['point_type'] = 'float'

            f.create_dataset('train', train.shape, dtype=train.dtype)[:] = train
            f.create_dataset('test', test.shape, dtype=test.dtype)[:] = test
            # Write the id buckets so on ingestion we will know what data to assign for each id.
            for j, id_bucket in enumerate(buckets):
                np_bucket = np.array(id_bucket, dtype=np.int32)
                f.create_dataset(f'{bucket_names[j]}_ids', np_bucket.shape, dtype=np_bucket.dtype)[:] = np_bucket

            neighbors = f.create_dataset(f'neighbors', (len(test), count), dtype='i')
            distances = f.create_dataset(f'distances', (len(test), count), dtype='f')
            
            # Generate ground truth only for the relevan bucket.
            train_bucket = np.array(bucket, dtype = np.int32)
            train_set = train[bucket]
            print(train_set.shape)
            bf = BruteForceBLAS(distance, precision=train.dtype)
            bf.fit(train_set)
            Parallel(n_jobs=multiprocessing.cpu_count(),  require='sharedmem')(delayed(calc_i)(i, x, bf, test, neighbors, distances, count, train_bucket) for i, x in enumerate(test))
            print(neighbors[0])
            print(distances[0])


if __name__ == "__main__":
    create_ds()