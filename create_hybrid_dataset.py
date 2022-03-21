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
import scipy.spatial

def calc_i(i, x, bf, test, neighbors, distances, count, orig_ids):
    if i % 1000 == 0:
        print('%d/%d...' % (i, len(test)))
    res = list(bf.query_with_distances(x, count))
    res.sort(key=lambda t: t[-1])
    neighbors[i] = [orig_ids[j] for j, _ in res]
    distances[i] = [d for _, d in res]

def create_buckets(train):
    bucket_0_5 = []
    bucket_1 = []
    bucket_2 = []
    bucket_5 = []
    bucket_10 = []
    bucket_20 = []
    bucket_50 = []
    other_bucket = []
    buckets = [bucket_0_5, bucket_1, bucket_2, bucket_5, bucket_10, bucket_20, bucket_50, other_bucket]
    bucket_names=['0.5', '1', '2', '5', '10', '20', '50', 'other']
    for i in range(train.shape[0]):
        if i % 200 == 19:      # 0.5%
            bucket_0_5.append(i)
        elif i % 100 == 17:    # 1%
            bucket_1.append(i)
        elif i % 50 == 9:     # 2%
            bucket_2.append(i)
        elif i % 20 == 7:     # 5%
            bucket_5.append(i)
        elif i % 10 == 3:     # 10%
            bucket_10.append(i)
        elif i % 2 == 0:      # 50%
            bucket_50.append(i)
        elif i % 5 <= 1:      # 20%
            bucket_20.append(i)
        else:
            other_bucket.append(i)
    print(len(bucket_0_5), len(bucket_1), len(bucket_2), len(bucket_5), len(bucket_10), len(bucket_20), len(bucket_50), len(other_bucket))
    numeric_values = {}
    text_values = {}
    for i, bucket_name in enumerate(bucket_names):
        numeric_values[bucket_name] = i
        text_values[bucket_name] = f'text_{i}'
    print(numeric_values)
    print(text_values)
    return buckets, bucket_names, numeric_values, text_values

@click.command()
@click.option('--data_set', type=click.Choice(DATASETS.keys(), case_sensitive=False), default='glove-100-angular')
@click.option('--percentile', type=click.Choice(['0.5', '1', '2', '5', '10', '20', '50'], case_sensitive=False), default=None)
def create_ds(data_set, percentile):
    ds, dimension= get_dataset(data_set)
    train = ds['train']
    test = ds['test']
    distance = ds.attrs['distance']
    count=len(ds['neighbors'][0])
    print(count)
    print(train.shape)
    buckets, bucket_names, numeric_values, text_values = create_buckets(train)

    if percentile is not None:
        i = ['0.5', '1', '2', '5', '10', '20', '50'].index(percentile)
        bucket = buckets[i]
        fn=f'{data_set}-hybrid-{bucket_names[i]}.hdf5'
        with h5py.File(fn, 'w') as f:
            f.attrs['type'] = 'dense'
            f.attrs['distance'] = ds.attrs['distance']
            f.attrs['dimension'] = len(test[0])
            f.attrs['point_type'] = 'float'
            f.attrs['bucket_names'] = bucket_names
            f.attrs['selected_bucket'] = bucket_names[i]
            for bucket_name in bucket_names:
                grp = f.create_group(bucket_name)
                grp["text"] = text_values[bucket_name]
                grp["number"] = numeric_values[bucket_name]

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
            train_set = np.empty((len(bucket), train.shape[1]), dtype=np.float32)
            for id in range(len(bucket)):
                train_set[id] = train[bucket[id]]
            bf = BruteForceBLAS(distance, precision=train.dtype)
            bf.fit(train_set)
            Parallel(n_jobs=multiprocessing.cpu_count(),  require='sharedmem')(delayed(calc_i)(i, x, bf, test, neighbors, distances, count, train_bucket) for i, x in enumerate(test))

    else:
        for i, bucket in enumerate(buckets):
            fn=f'{data_set}-hybrid-{bucket_names[i]}.hdf5'
            with h5py.File(fn, 'w') as f:
                f.attrs['type'] = 'dense'
                f.attrs['distance'] = ds.attrs['distance']
                f.attrs['dimension'] = len(test[0])
                f.attrs['point_type'] = 'float'
                f.attrs['bucket_names'] = bucket_names
                f.attrs['selected_bucket'] = bucket_names[i]
                for bucket_name in bucket_names:
                    grp = f.create_group(bucket_name)
                    grp["text"] = text_values[bucket_name]
                    grp["number"] = numeric_values[bucket_name]

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
                train_set = np.empty((len(bucket), train.shape[1]), dtype=np.float32)
                for id in range(len(bucket)):
                    train_set[id] = train[bucket[id]]
                print(train_set.shape)
                bf = BruteForceBLAS(distance, precision=train.dtype)
                bf.fit(train_set)
                Parallel(n_jobs=multiprocessing.cpu_count(),  require='sharedmem')(delayed(calc_i)(i, x, bf, test, neighbors, distances, count, train_bucket) for i, x in enumerate(test))
                print(neighbors[1])
                print(distances[1])


if __name__ == "__main__":
    create_ds()
