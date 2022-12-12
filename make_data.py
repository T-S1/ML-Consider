import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf

SEED = 29
tf.random.set_seed(SEED)

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    x, y = make_regression(
        n_samples=12000,
        n_features=100,
        n_informative=10,
        n_targets=1,
        noise=0.5,
        random_state=SEED
    )

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=1000, random_state=SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, random_state=SEED
    )

    np.save("./data/x_train.npy", x_train)
    np.save("./data/x_val.npy", x_val)
    np.save("./data/x_test.npy", x_test)
    np.save("./data/y_train.npy", y_train)
    np.save("./data/y_val.npy", y_val)
    np.save("./data/y_test.npy", y_test)


if __name__ == "__main__":
    main()
