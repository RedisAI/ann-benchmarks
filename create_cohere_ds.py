from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS
import numpy as np
import click
import h5py
from joblib import Parallel, delayed
import multiprocessing
import tqdm
from datasets import load_dataset


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


def train_test_split(X, test_size=10000, dimension=None):
    import sklearn.model_selection

    if dimension is None:
        dimension = X.shape[1]
    print("Splitting %d*%d into train/test" % (X.shape[0], dimension))
    return sklearn.model_selection.train_test_split(
        X, test_size=test_size, random_state=1
    )


@click.command()
@click.option("--train_size", default=1000000, help="Train size.")
@click.option("--test_size", default=10000, help="Test size.")
@click.option("--distance", default="angular", help="distance metric.")
def create_ds(train_size, test_size, distance):
    dim = 768
    total_vecs = train_size + test_size
    X = np.zeros((total_vecs, dim), dtype=np.float32)
    data = load_dataset(
        "Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True
    )
    pos = 0
    print(
        f"generating train set of size {train_size} and test set of size {test_size}. Fetching {total_vecs} embeddings."
    )
    pbar = tqdm.tqdm(total=total_vecs)
    for row in data:
        emb = row["emb"]
        v = [float(x) for x in emb]
        X[pos] = np.array(v, dtype=np.float32)
        pbar.update(1)
        pos = pos + 1
        if pos >= total_vecs:
            break

    X_train, X_test = train_test_split(X, test_size)

    human_size = human_format(train_size)
    write_output(
        train=np.array(X_train),
        test=np.array(X_test),
        fn=f"cohere-{dim}-{human_size}-{distance}.hdf5",
        distance=distance,
        point_type="float",
        count=100,
    )


if __name__ == "__main__":
    create_ds()
