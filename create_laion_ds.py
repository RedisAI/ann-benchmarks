from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS
import numpy as np
import click
import h5py
from joblib import Parallel, delayed
import multiprocessing
import tqdm
from urllib.request import urlretrieve
import sklearn.model_selection
import os
import wget
import sys


def download(src, dst):
    if not os.path.exists(dst):
        # TODO: should be atomic
        print("downloading %s -> %s..." % (src, dst))
        wget.download(src, dst)


def calc_i(i, x, bf, test, neighbors, distances, count):
    if i % 1000 == 0:
        print("%d/%d..." % (i, len(test)))
    res = list(bf.query_with_distances(x, count))
    res.sort(key=lambda t: t[-1])
    neighbors[i] = [j for j, _ in res]
    distances[i] = [d for _, d in res]


def calc(bf, test, neighbors, distances, count):
    Parallel(n_jobs=multiprocessing.cpu_count(), require="sharedmem")(
        delayed(calc_i)(i, x, bf, test, neighbors, distances, count)
        for i, x in enumerate(test)
    )


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return "%.0f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


def write_output(train, test, fn, distance, point_type="float", count=100):
    f = h5py.File(fn, "w")
    f.attrs["type"] = "dense"
    f.attrs["distance"] = distance
    f.attrs["dimension"] = len(train[0])
    f.attrs["point_type"] = point_type
    print("train size: %9d * %4d" % train.shape)
    print("test size:  %9d * %4d" % test.shape)
    f.create_dataset("train", (len(train), len(train[0])), dtype=train.dtype)[:] = train
    f.create_dataset("test", (len(test), len(test[0])), dtype=test.dtype)[:] = test
    neighbors = f.create_dataset("neighbors", (len(test), count), dtype="i")
    distances = f.create_dataset("distances", (len(test), count), dtype="f")
    bf = BruteForceBLAS(distance, precision=train.dtype)

    bf.fit(train)
    calc(bf, test, neighbors, distances, count)
    f.close()


@click.command()
@click.option("--train_size", default=1000000, help="Train size.")
@click.option("--test_size", default=10000, help="Test size.")
@click.option("--distance", default="angular", help="distance metric.")
def create_ds(train_size, test_size, distance):
    dim = 512
    total_vecs = train_size + test_size
    file_limit = 409
    vector_limit = 400 * 1000000
    if total_vecs > vector_limit:
        print("vector limit is larger than the dataset")
        sys.exit(1)
    pos = 0
    print(
        f"generating train set of size {train_size} and test set of size {test_size}. Fetching {total_vecs} embeddings."
    )
    X = np.zeros((total_vecs, dim), dtype=np.float32)

    pbar = tqdm.tqdm(total=total_vecs)
    file_n = 0
    while pos < total_vecs:
        filename = f"img_emb_{file_n}.npy"
        url = f"https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings/images/{filename}"
        download(url, filename)
        img_emb = np.load(filename)
        for row in img_emb:
            X[pos] = row.astype(np.float32)
            pbar.update(1)
            pos = pos + 1
            if pos >= total_vecs:
                break
        file_n = file_n + 1
        if file_n > file_limit:
            print("vector limit is larger than the dataset")
            sys.exit(1)

    print("Splitting %d*%d into train/test" % (X.shape[0], X.shape[1]))
    X_train, X_test = sklearn.model_selection.train_test_split(
        X, test_size=test_size, random_state=1
    )

    human_size = human_format(train_size)
    write_output(
        train=np.array(X_train),
        test=np.array(X_test),
        fn=f"laion-img-emb-{dim}-{human_size}-{distance}.hdf5",
        distance=distance,
        point_type="float",
        count=100,
    )


if __name__ == "__main__":
    create_ds()
