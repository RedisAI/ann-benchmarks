import os
import pathlib

from ann_benchmarks.main import main
from multiprocessing import freeze_support

if __name__ == "__main__":
    workdir = pathlib.Path(__file__).parent.absolute()
    print("Changing the workdir to {}".format(workdir))
    os.chdir(workdir)
    freeze_support()
    main()
