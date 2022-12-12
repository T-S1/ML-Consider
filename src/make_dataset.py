import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf

SEED = 29
tf.random.set_seed(SEED)

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np


class DatasetMaker():
    def __init__(self, name: str):
        target_dir = f"./data/raw/{name}"
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        self.x_train_path = f"{target_dir}/x_train.npy"
        self.x_val_path = f"{target_dir}/x_val.npy"
        self.x_test_path = f"{target_dir}/x_test.npy"
        self.y_train_path = f"{target_dir}/y_train.npy"
        self.y_val_path = f"{target_dir}/y_val.npy"
        self.y_test_path = f"{target_dir}/y_test.npy"

        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.is_done = False

        if os.path.isfile(self.y_test_path):
            self.x_train = np.load(self.x_train_path)
            self.x_val = np.load(self.x_val_path)
            self.x_test = np.load(self.x_test_path)
            self.y_train = np.load(self.y_train_path)
            self.y_val = np.load(self.y_val_path)
            self.y_test = np.load(self.y_test_path)
            self.is_done = True

    def run(self):
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

        np.save(self.x_train_path, x_train)
        np.save(self.x_val_path, x_val)
        np.save(self.x_test_path, x_test)
        np.save(self.y_train_path, y_train)
        np.save(self.y_val_path, y_val)
        np.save(self.y_test_path, y_test)

        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_val_data(self):
        return self.x_val, self.y_val

    def get_test_data(self):
        return self.x_test, self.y_test
