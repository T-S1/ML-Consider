import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf

SEED = 29
tf.random.set_seed(SEED)

import pickle
import numpy as np
import tensorflow.keras as keras


class ModelBuilder():
    def __init__(self, name: str):
        target_dir = f"./models/{name}"
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        self.structure = None
        self.model = None
        self.df_hisory = None

        self.structure_path = f"{target_dir}/structure.pkl"
        self.weights_path = f"{target_dir}/weights.h5"
        self.df_hisory_path = f"{target_dir}/df_history.pkl"
        self.is_done = False

        if os.path.isfile(self.hisory_path):
            with open(self.structure_path, "rb") as f:
                self.structure = pickle.load(f)
            model = build_model()

    def build_model(self):
        
