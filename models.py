'''
Object classes for models.
Contains methods to compute predictions and losses on a batch of data.
'''

import tensorflow as tf
import os
import csv
import numpy as np
import json
import tflearn
import time
from matplotlib import cm
import matplotlib.pyplot as plt
plt.style.use('bmh')

class Model:
    def __init__(self, nFeatures, nOutputs, sess, model, nGdIter):
        self.nFeatures = nFeatures
        self.nOutputs = nOutputs
        self.sess = sess
        self.model = model

        self.ftrue = tf.placeholder(tf.float32, shape=[None, nOutputs], name='trueF')

        self.x_ = tf.placeholder(tf.float32, shape=[None, 1], name='x')
        self.y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y')
        self.u_ = tf.placeholder(tf.float32, shape=[None, 1], name='u') # precomputed sign of u
        self.v_ = tf.placeholder(tf.float32, shape=[None, 2], name='v')

        if model == 'picnn':
            f = self.f_picnn
        else:
            raise NotImplementedError

        self.fest_ = f(self.x_, self.y_, self.v_, self.u_)

        self.mse_ = tf.reduce_mean(tf.square(self.fest_ - self.ftrue))

        self.opt = tf.train.AdamOptimizer(0.001)
        self.theta_ = tf.trainable_variables()
        self.gv_ = [(g,v) for g,v in
                    self.opt.compute_gradients(self.mse_, self.theta_)
                    if g is not None]
        self.train_step = self.opt.apply_gradients(self.gv_)

        self.theta_cvx_ = [v for v in self.theta_
                           if 'proj' in v.name and 'W:' in v.name]

        self.makeCvx = [v.assign(tf.abs(v)/10.) for v in self.theta_cvx_]
        self.proj = [v.assign(tf.maximum(v, 0)) for v in self.theta_cvx_]

        self.theta_mono_ = [v for v in self.theta_
                           if 'pass' in v.name and 'W:' in v.name]
        self.passthro = [v.assign(tf.minimum(v, 0)) for v in self.theta_mono_]

        # for g,v in self.gv_:
        #     variable_summaries(g, 'gradients/'+v.name)

        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=0)

    def train(self, args, dataX, dataY, validation_data):
        save = os.path.join(os.path.expanduser(args.save),
                            "{}.{}".format(args.model, args.dataset))
        validationX, validationY = validation_data
        nTrain = dataX.shape[0]

        imgDir = os.path.join(save, 'imgs')
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)

        trainFields = ['iter', 'loss', 'validate_loss']
        trainF = open(os.path.join(save, 'train.csv'), 'w')
        trainW = csv.writer(trainF)
        trainW.writerow(trainFields)

        self.trainWriter = tf.summary.FileWriter(os.path.join(save, 'train'), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.makeCvx)

        nParams = np.sum(v.get_shape().num_elements() for v in tf.trainable_variables())

        meta = {'nTrain': nTrain, 'nParams': nParams, 'nEpoch': args.nEpoch}
        metaP = os.path.join(save, 'meta.json')
        with open(metaP, 'w') as f:
            json.dump(meta, f, indent=2)

        '''precompute sign of u before feeding it to the network'''
        ui_sign = np.where(dataX[:,4] % 2 == 0, 1, -1)
        ui_sign = ui_sign.astype(np.float32).reshape((-1,1))
        ui_val_sign = np.where(validationX[:,4] % 2 == 0, 1, -1)
        ui_val_sign = ui_val_sign.astype(np.float32).reshape((-1,1))

        '''training loop'''
        bestMSE = None
        for i in range(args.nEpoch):
            tflearn.is_training(True)

            print("=== Epoch {} ===".format(i))
            start = time.time()

            _, trainMSE, yn = self.sess.run(
                [self.train_step, self.mse_, self.y_],
                feed_dict={self.x_: dataX[:,0].reshape((-1,1)), self.y_: dataX[:,1].reshape((-1,1)), self.u_: ui_sign, self.v_: dataX[:,2:4].reshape((-1,2)), self.ftrue: dataY})
            if len(self.proj) > 0:
                self.sess.run(self.proj)
            self.sess.run(self.passthro)

            validationMSE = self.sess.run(
                self.mse_,
                feed_dict={self.x_: validationX[:,0].reshape((-1,1)), self.y_: validationX[:,1].reshape((-1,1)), self.u_: ui_val_sign, self.v_: validationX[:,2:4].reshape((-1,2)), self.ftrue: validationY})


            trainW.writerow((i, trainMSE, validationMSE))
            trainF.flush()

            print(" + loss: {:0.5e}".format(trainMSE))
            print(" + vali: {:0.5e}".format(validationMSE))
            print(" + time: {:0.2f} s".format(time.time()-start))

        trainF.close()
        self.plot_loss(args)

    def f_picnn(self, x, y, v, ui, reuse=False):
        fc = tflearn.fully_connected
        xy = tf.concat((x, y),1)
        ui_2d = tf.tile(ui, [1, 200])

        with tf.variable_scope('vc') as s:
            vc = fc(v, 1, scope=s, reuse=reuse, bias=True)
            vc = tf.nn.relu(vc)

        prevZ, prevU = None, x
        prevU = tf.concat([prevU, vc], 1)
        
        for layerI, sz in enumerate([200, 200, 1]):

            if sz != 1:
                with tf.variable_scope('u'+str(layerI)) as s:
                    u = fc(prevU, sz, scope=s, reuse=reuse)
                    u = tf.nn.relu(u)

            z_add = []

            if prevZ is not None:
                with tf.variable_scope('z{}_zu_u'.format(layerI)) as s:
                    prevU_sz = prevU.get_shape()[1].value
                    zu_u = fc(prevU, prevU_sz, reuse=reuse, scope=s,
                            activation='relu', bias=True)
                with tf.variable_scope('z{}_zu_proj'.format(layerI)) as s:
                    zu_mul = tf.multiply(prevZ, zu_u)
                    if sz ==1: zu_mul = tf.multiply(zu_mul, ui_2d)
                    z_zu = fc(zu_mul, sz, reuse=reuse, scope=s, bias=False)
                z_add.append(z_zu)

            with tf.variable_scope('z{}_yu_u'.format(layerI)) as s:
                yu_u = fc(prevU, self.nOutputs, reuse=reuse, scope=s, bias=True)
            with tf.variable_scope('z{}_yu_pass'.format(layerI)) as s:
                yu_mul = tf.multiply(y, yu_u)
                # if sz ==1: yu_mul = tf.multiply(yu_mul, ui_2d)
                z_yu = fc(yu_mul, sz, reuse=reuse, scope=s, bias=False)
            z_add.append(z_yu)

            with tf.variable_scope('z{}_u'.format(layerI)) as s:
                z_u = fc(prevU, sz, reuse=reuse, scope=s, bias=True)
            z_add.append(z_u)

            z = tf.add_n(z_add)
            if sz != 1:
                z = tf.nn.relu(z)

            prevU = u
            prevZ = z
    
        return tf.contrib.layers.flatten(z)

    def plot_loss(self, args):
        save = os.path.join(os.path.expanduser(args.save),
                            "{}.{}".format(args.model, args.dataset))
        imgDir = os.path.join(save, 'imgs')
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)

        train_loss = []
        validation_loss = []
        with open(os.path.join(save, 'train.csv'), 'r') as f:
            reader = csv.reader(f)
            next(reader)  
            for row in reader:
                if len(row) == 0:
                    continue
                train_loss.append(float(row[1]))
                validation_loss.append(float(row[2]))

        plt.figure()
        plt.plot(train_loss, label='Training Loss')
        plt.plot(validation_loss, label='Validation Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(imgDir, 'convergence_plot.png'))

        plt.figure()
        plt.loglog(train_loss, label='Training Loss')
        plt.loglog(validation_loss, label='Validation Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(imgDir, 'convergence_loglog_plot.png'))

def variable_summaries(var, name=None):
    if name is None:
        name = var.name
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stdev'):
            stdev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stdev/' + name, stdev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)