"""Waiting for a good D->Python wrapper...."""

from itertools import chain
import os
import shutil
import subprocess
import tempfile

import numpy as np


class Model(object):

    def __init__(self, bits, loss):
        self.intercept = 0.0
        self.weights = np.zeros(1 << bits)
        self.loss = loss

    def predict(x):
        res = self.intercept
        for feat, val in x:
            feat_hash = feat #TODO: implement same hasher...
            res += self.weights[feat] * val
        return res

class SqPUG(object):

    def __init__(self):
        self._folder = tempfile.mkdtemp(prefix="sqpug_")

    def __del__(self):
        shutil.rmtree(self._folder, ignore_errors=True)

    @staticmethod
    def dump_data(X, Y, path, format="sparse"):

        assert len(X) == len(Y), ("Feature matrix and Label vector should "
            "have the same size.")

        with open(path, "w") as f:
            for x, y in zip(X, Y):
                feats_str = " ".join("%s:%s" % (key, str(val))
                                     for key, val in x)
                line = "%s | %s\n" % (str(y), feats_str)
                f.write(line)

    @staticmethod
    def load_model(model_path):
        with open(model_path, "r") as f:
            # Number of bits for hashing trick:
            line = f.readline().rstrip()
            bits = int(line.split(":")[1])

            # LossType:
            line = f.readline().rstrip()
            loss = line.split(":")[1]

            model = Model(bits, loss)

            # Intercept:
            line = f.readline().rstrip()
            model.intercept = float(line.split(":")[1])

            # Model Weights:
            for line in f.readlines():
                tokens = line.rstrip().split(":")
                ind, val = int(tokens[0]), float(tokens[1])
                model.weights[ind] = val
        return model

    def learn(self, X, Y, **kwargs):

        tmp_data_path = os.path.join(self._folder, "data.in")
        self.dump_data(X, Y, tmp_data_path)

        tmp_model_out = os.path.join(self._folder, "model.out")

        mu = kwargs.pop("mu", None)
        if mu:
            kwargs["lambda"] = mu

        kw_list = list(chain.from_iterable(("--%s" % k, str(v))
                                       for (k,v) in kwargs.iteritems()))
        call_list = [
            "sqpug",
            "--data", tmp_data_path,
            "--model_out", tmp_model_out]
        call_list.extend(kw_list)

        p = subprocess.Popen(call_list)
        p.wait()

        self.model = self.load_model(tmp_model_out)
