import gzip

import numpy as np
import tensorflow as tf
from input_data import read_data_sets

# Load the dataset
# f = gzip.open('../../data/mnist.pkl.gz', 'rb')
# train_set, valid_set, test_set = cPickle.load(f)
# f.close()

print(read_data_sets())
