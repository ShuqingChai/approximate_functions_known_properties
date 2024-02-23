import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import numpy.random as npr

np.set_printoptions(precision=2)
np.seterr(all='raise')

import argparse
# import sys
# import pickle
import shutil
# from datetime import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

# from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from models import Model
from samplers import OnlineSampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='work')
    parser.add_argument('--nEpoch', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default="picnn",
                        choices=['picnn'])
    parser.add_argument('--dataset', type=str, default="linear",
                        choices=['linear', 'multiply'])
    parser.add_argument('--noncvx', type=bool, default=True)
    parser.add_argument("--noise", type=str, default = "no_noise", choices=["no_noise", "fixed_variance"])
    parser.add_argument("--bounds", type=float, nargs=4, default=[0, 1, 0, 1])
    parser.add_argument("--num_data", type=int, default=500)

    args = parser.parse_args()

    npr.seed(args.seed)
    tf.set_random_seed(args.seed)

    save = os.path.join(os.path.expanduser(args.save),
                        "{}.{}".format(args.model, args.dataset))
    if os.path.isdir(save):
        shutil.rmtree(save)
    os.makedirs(save, exist_ok=True)

    # prepare data
    sampler_linear = OnlineSampler(args.dataset, args.num_data, args.bounds, args.noise, args.noncvx)
    dataX, dataY = sampler_linear.sample()
    trainX, validationX, trainY, validationY = train_test_split(dataX, dataY, test_size=0.2, random_state=args.seed)
    trainY = sampler_linear.pollute_observation(trainY)

    nData = dataX.shape[0]
    nFeatures = dataX.shape[1]
    nOutputs = 1
    nXy = nFeatures + nOutputs

    config = tf.ConfigProto() #log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = Model(nFeatures, nOutputs, sess, args.model, nGdIter=30)
        model.train(args, trainX, trainY, (validationX, validationY))

if __name__=='__main__':
    main()
