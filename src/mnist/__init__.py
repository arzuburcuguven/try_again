import os

_MNIST_ROOT = os.path.dirname(__file__)  # root of test folder
_SRC_ROOT = os.path.dirname(_MNIST_ROOT)  # root of project
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data
_PATH_DATA_RAW = os.path.join(_PROJECT_ROOT, "data/raw/corruptmnist")  # root of data
_PATH_DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data/processed")  # root of data
