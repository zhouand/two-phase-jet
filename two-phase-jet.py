#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""



import numpy as np
import time
from pyDOE import lhs
import matplotlib
# matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import shutil
import pickle
import scipy.io
import random

# Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # CPU:-1; GPU0: 1; GPU1: 0;


random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)


class SAT:
    # Initialize the class
    def __init__(self, Collo, sat_c_1, uv_layers_sat, lb, ub, ExistModelsat=0, uvDir=''):

        # Count for callback function
        self.count=0

        # Bounds
        self.lb = lb
        self.ub = ub
        # Mat. properties
        #self.rho = 1.138
        #self.cp = 1006.43

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]
        #self.p_c_1 = p_c_1
        #self.tem_c_1 = tem_c_1
        self.sat_c_1 = sat_c_1

        # Define layers
        self.uv_layers_sat = uv_layers_sat

        # Initialize NNs
        if ExistModelsat == 0 :
            self.uv_weights, self.uv_biases = self.initialize_NN_sat(self.uv_layers_sat)
        else:
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN_sat(uvDir, self.uv_layers_sat)

        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])
        #self.p_c_tf = tf.placeholder(tf.float32, shape=[None, self.p_c_1.shape[1]])
        #self.e_c_tf = tf.placeholder(tf.float32, shape=[None, self.e_c_1.shape[1]])
        self.sat_c_tf = tf.placeholder(tf.float32, shape=[None, self.sat_c_1.shape[1]])

        # tf graphs
        self.sat_pred = self.net_uv_sat(self.x_c_tf, self.y_c_tf)

    
        self.loss = tf.reduce_mean(tf.square(self.sat_c_tf - self.sat_pred))\

        # Optimizer for solution
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.uv_weights + self.uv_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 100000,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1*np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.uv_weights + self.uv_biases)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN_sat(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init_sat(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init_sat(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def save_NN_sat(self, fileDir):

        uv_weights = self.sess.run(self.uv_weights)
        uv_biases = self.sess.run(self.uv_biases)

        with open(fileDir, 'wb') as f:
            pickle.dump([uv_weights, uv_biases], f)
            print("Save uv NN parameters successfully...")

    def load_NN_sat(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases = pickle.load(f)

            # Stored model must has the same # of layers
            assert num_layers == (len(uv_weights)+1)

            for num in range(0, num_layers - 1):
                W = tf.Variable(uv_weights[num])
                b = tf.Variable(uv_biases[num])
                weights.append(W)
                biases.append(b)
                print(" - Load NN parameters successfully...")
        return weights, biases

    def neural_net_sat(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        #H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv_sat(self, x, y):
        sat = self.neural_net_sat(tf.concat([x, y], 1), self.uv_weights, self.uv_biases)
        return sat


    def train_sat(self, iter, learning_rate):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.sat_c_tf: self.sat_c_1,
                   self.learning_rate: learning_rate}
        loss = []

        for it in range(iter):

            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e' %
                      (it, loss_value))

            loss.append(self.sess.run(self.loss, tf_dict))

        return loss
    
    def callback(self, loss):
        self.count = self.count+1
        print('{} th iterations, Loss: {}'.format(self.count, loss))
        
    def train_bfgs_sat(self):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,self.sat_c_tf: self.sat_c_1}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict_sat(self, x_star, y_star):
        sat_star = self.sess.run(self.sat_pred, {self.x_c_tf: x_star, self.y_c_tf: y_star})
        return sat_star

    def getloss_sat(self):  # To be updated

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.sat_c_tf: self.sat_c_1}

        loss = self.sess.run(self.loss, tf_dict)

        return loss



class PINN_laminar_flow:
    # Initialize the class
    def __init__(self, Collo, u_c, v_c, p_c, rho_c, e_c, sat_c_1, uv_layers, lb, ub, ExistModel=0, uvDir=''):

        # Count for callback function
        self.count=0

        # Bounds
        self.lb = lb
        self.ub = ub

        # Mat. properties
        #self.rho = 1.138
        #self.cp = 1006.43

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]
        
        
        self.u_c_1 = u_c
        self.v_c_1 = v_c
        self.p_c_1 = p_c
        self.e_c_1 = e_c
        self.rho_c_1 = rho_c


        # Define layers
        self.uv_layers = uv_layers

        # Initialize NNs
        if ExistModel== 0 :
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)
        else:
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, self.uv_layers)

        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])
        self.u_c_tf = tf.placeholder(tf.float32, shape=[None, self.u_c_1.shape[1]])
        self.v_c_tf = tf.placeholder(tf.float32, shape=[None, self.v_c_1.shape[1]])
        self.p_c_tf = tf.placeholder(tf.float32, shape=[None, self.p_c_1.shape[1]])
        self.e_c_tf = tf.placeholder(tf.float32, shape=[None, self.e_c_1.shape[1]])
        self.rho_c_tf = tf.placeholder(tf.float32, shape=[None, self.rho_c_1.shape[1]])

        # tf graphs
        self.u_pred, self.v_pred, self.p_pred, self.e_pred, self.rho_pred = self.net_uv(self.x_c_tf, self.y_c_tf)
        self.f_pred_eq, self.f_pred_u, self.f_pred_v, self.f_pred_e, self.f_pred_eos\
            = self.net_f(self.x_c_tf, self.y_c_tf)

    
        self.loss_f = tf.reduce_mean(tf.square(self.u_c_tf - self.u_pred))\
                      + tf.reduce_mean(tf.square(self.v_c_tf - self.v_pred))\
                      + 20*tf.reduce_mean(tf.square(self.p_c_tf - self.p_pred))\
                      + tf.reduce_mean(tf.square(self.e_c_tf - self.e_pred))
        self.loss_eq = tf.reduce_mean(tf.square(self.f_pred_eos))\
                      + tf.reduce_mean(tf.square(self.f_pred_eq))\
                      + tf.reduce_mean(tf.square(self.f_pred_u))\
                      + tf.reduce_mean(tf.square(self.f_pred_v))\
                      + tf.reduce_mean(tf.square(self.f_pred_e))

        #self.exp = tf.reduce_mean(tf.square(self.exp_real - self.exp_pred))
        # Coefficients could affect the accuracy and convergence of the result
        self.loss = self.loss_f + self.loss_eq/100

        # Optimizer for solution
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.uv_weights + self.uv_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 100000,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1*np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)#,var_list=self.uv_weights + self.uv_biases)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def save_NN(self, fileDir):

        uv_weights = self.sess.run(self.uv_weights)
        uv_biases = self.sess.run(self.uv_biases)

        with open(fileDir, 'wb') as f:
            pickle.dump([uv_weights, uv_biases], f)
            print("Save uv NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases = pickle.load(f)

            # Stored model must has the same # of layers
            assert num_layers == (len(uv_weights)+1)

            for num in range(0, num_layers - 1):
                W = tf.Variable(uv_weights[num])
                b = tf.Variable(uv_biases[num])
                weights.append(W)
                biases.append(b)
                print(" - Load NN parameters successfully...")
        return weights, biases



    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X#2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H =  tf.tanh(5*a[l]*tf.add(tf.matmul(H, W), b))\
                + 5*a1[l]*tf.sin(10*F1[l]*tf.add(tf.matmul(H, W), b))\
                + 5*a2[l]*tf.sin(20*F2[l]*tf.add(tf.matmul(H, W), b))\
                    + 5*a3[l]*tf.sin(30*F3[l]*tf.add(tf.matmul(H, W), b))\
                        + 5*a4[l]*tf.tanh(40*F4[l]*tf.add(tf.matmul(H, W), b))\
                            + 5*a5[l]*tf.tanh(50*F5[l]*tf.add(tf.matmul(H, W), b))\
                                + 5*a6[l]*tf.tanh(60*F6[l]*tf.add(tf.matmul(H, W), b))\
                                    + 5*a7[l]*tf.tanh(70*F7[l]*tf.add(tf.matmul(H, W), b))\
                                        + 5*a8[l]*tf.tanh(80*F8[l]*tf.add(tf.matmul(H, W), b))\
                                            + 5*a9[l]*tf.tanh(90*F9[l]*tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_uv(self, x, y):
        psips = self.neural_net(tf.concat([x, y], 1), self.uv_weights, self.uv_biases)
        psi = psips[:, 0:1]
        p = psips[:, 1:2]
        e = psips[:, 2:3]
        rho = psips[:, 3:4]
        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]
        return u, v, p, rho, e
    
    def net_f(self, x, y):

        #rho_l = com_rho_l(tem)
        e_real = np.array([])
        
        u, v, p, rho, e = self.net_uv(x, y)
        #p_real = 8.134*rho*#eos(tem,p)
        
        u = ((u*(max(u_c)-min(u_c))+min(u_c)))/400
        v = ((v*(max(v_c)-min(v_c))+min(v_c)))/400
        p = ((p*(max(p_c)-min(p_c))+min(p_c)))/(400**2 * 0.41567364)
        rho = (rho*(max(rho_c)-min(rho_c))+min(rho_c))/0.41567364
        e = ((e*(max(e_c)-min(e_c))+min(e_c)))/(400**2 * 0.41567364)
        sat = model_sat.predict_sat(XY_c[:,0:1],XY_c[:,1:2])
        plt.scatter(XY_c[:,0:1],XY_c[:,1:2],c=sat)
        sat = (sat*(max(sat_c)-min(sat_c))+min(sat_c))
        e_int = e
        e_real = 4.5481*p**3 - 12.29*p**2 + 13.03*p + 32.393
        
        e = rho*(e + (u**2+v**2)/2)
        uu = u*u
        vv = v*v
        rho_u = rho*u
        rho_v = rho*v
        rho_uu = rho*u*u
        rho_uv = rho*u*v
        rho_e = rho*e
        rho_eu = rho*e*u
        e_u = e*u
        e_v = e*v
        p_u = p*u
        p_v = p*v
        rho_vv = rho*v*v
        rho_ev = rho*e*v

        S = sat  * (-p + p_sat)/p_sat * (900/(400 * 0.41567364))
        h = 2200*1e3/(400**2 * 0.41567364)


        # Plane stress problem
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        uu_x = tf.gradients(uu, x)[0]
        vv_y = tf.gradients(vv, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        

        rho_x = tf.gradients(rho, x)[0]
        rho_y = tf.gradients(rho, y)[0]

        e_x = tf.gradients(e, x)[0]
        e_y = tf.gradients(e, y)[0]
        e_v_y = tf.gradients(e_v, y)[0]
        e_u_x = tf.gradients(e_u, x)[0]
        

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        p_u_x = tf.gradients(p_u, x)[0]
        p_v_y = tf.gradients(p_v, y)[0]
        

        rho_e_u_x = tf.gradients(rho_eu, x)[0]
        
        rho_u_x = tf.gradients(rho_u, x)[0]
        rho_uu_x = tf.gradients(rho_uu, x)[0]
        rho_uv_x = tf.gradients(rho_uv, x)[0]
        
        
        rho_v_y = tf.gradients(rho_v, y)[0]
        rho_uv_y = tf.gradients(rho_uv, y)[0]
        rho_vv_y = tf.gradients(rho_vv, y)[0]
        rho_ev_y = tf.gradients(rho_ev, y)[0]
        rho_u_y = tf.gradients(rho_u, y)[0]

        f_eq = rho_u_x + rho_v_y - S
        f_e = e_u_x + p_u_x + e_v_y + p_v_y + h*S


        
        f_u = rho_uu_x + p_x + rho_uv_y 
        f_v = rho_uv_x + rho_vv_y + p_y
        f_eos = e - e_int

        return f_eq*10, f_u, f_v, f_e/10, f_eos

    def callback(self, loss):
        self.count = self.count+1
        print('{} th iterations, Loss: {}'.format(self.count, loss))


    def train(self, iter, learning_rate):
        loss_plt = np.array([])
        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.u_c_tf: self.u_c_1, self.v_c_tf: self.v_c_1, self.p_c_tf: self.p_c_1, self.rho_c_tf: self.rho_c_1, self.e_c_tf: self.e_c_1,
                   self.learning_rate: learning_rate}

        loss_f = []
        loss = []
        loss_eq = []
        loss_a = []

        for it in range(iter):

            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                loss = self.sess.run(self.loss, tf_dict)
                loss_a = self.sess.run(a, tf_dict)
                loss_plt = np.append(loss_plt,[it, loss])
                print('It: %d, Loss: %.3e' % 
                      (it, loss))


                loss_f.append(self.sess.run(self.loss_f, tf_dict))
                loss_eq.append(self.sess.run(self.loss_eq, tf_dict))


        return loss, loss_eq, loss_plt

    def train_bfgs(self):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.u_c_tf: self.u_c_1, self.v_c_tf: self.v_c_1, self.p_c_tf: self.p_c_1, self.rho_c_tf: self.rho_c_1, self.e_c_tf: self.e_c_1}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, x_star, y_star):
        u_star = self.sess.run(self.u_pred, {self.x_c_tf: x_star, self.y_c_tf: y_star})
        v_star = self.sess.run(self.v_pred, {self.x_c_tf: x_star, self.y_c_tf: y_star})
        p_star = self.sess.run(self.p_pred, {self.x_c_tf: x_star, self.y_c_tf: y_star})
        e_star = self.sess.run(self.e_pred, {self.x_c_tf: x_star, self.y_c_tf: y_star})
        return u_star, v_star, p_star, e_star

    def getloss(self):  # To be updated

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.u_c_tf: self.u_c_1, self.v_c_tf: self.v_c_1, self.p_c_tf: self.p_c_1, self.rho_c_tf: self.rho_c_1, self.e_c_tf: self.e_c_1}

        loss_f = self.sess.run(self.loss_f, tf_dict)
        loss = self.sess.run(self.loss, tf_dict)
        loss_eq = self.sess.run(self.loss_eq, tf_dict)
        loss_a = self.sess.run(a, tf_dict)

        return loss, loss_eq, loss_plt

def preprocess(dir):
    # Directory of reference solution
    data = scipy.io.loadmat(dir)

    X = data['x']
    Y = data['y']
    #T = data['t']
    Exact_u = data['u']
    Exact_v = data['v']
    Exact_p = data['p']
    Exact_rho = data['rho']
    Exact_e = data['e']
    Exact_l = data['l']
    #Exact_tem = data['t']
    
    x_star = X.flatten()[:, None]
    y_star = Y.flatten()[:, None]
    #t_star = T.flatten()[:, None]
    u_star = Exact_u.flatten()[:, None]
    v_star = Exact_v.flatten()[:, None]
    p_star = Exact_p.flatten()[:, None]
    rho_star = Exact_rho.flatten()[:, None]
    e_star = Exact_e.flatten()[:, None]
    l_star = Exact_l.flatten()[:, None]
    #tem_star = Exact_tem.flatten()[:, None]

    return x_star, y_star, u_star, v_star, p_star, rho_star, e_star, l_star

def postProcess(xmin, xmax, ymin, ymax, field, s=2):
    ''' num: Number of time step
    '''
    [x_pred, y_pred, p_pred,sat_pred] = field

    # fig, axs = plt.subplots(2)
    fig, ax = plt.subplots(nrows=2, figsize=(6, 10))
    # fig.subplots_adjust(hspace=0.2, wspace=0.2)

    
    cf = ax[0].scatter(x_pred, y_pred, c=p_pred, alpha=0.7, edgecolors='none', cmap='rainbow', vmin=0,vmax=0.1,marker='o', s=s)
    ax[0].axis('square')
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])

    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0].set_title('p error')
    cb2=fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)
    cb2.set_ticks([0,0.025,0.05,0.075,0.1])
    
    
    cf = ax[1].scatter(x_pred, y_pred, c=sat_pred, alpha=0.7, edgecolors='none', cmap='rainbow',vmin=0,vmax=0.015, marker='o', s=s)
    ax[1].axis('square')
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])

    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1].set_title('sat error')
    cb4=fig.colorbar(cf, ax=ax[1], fraction=0.046, pad=0.04)
    cb4.set_ticks([0,0.025,0.05])
    
    plt.savefig('./output-stedy/stedy-3900.tif',dpi=300)
    plt.close('all')

def hyper_parameters_A(size): 
    a = tf.Variable(tf.constant(0.2, shape=size))
    return a

def hyper_parameters_freq1(size):
    return tf.Variable(tf.constant(0.2, shape=size))

def hyper_parameters_freq2(size):
    return tf.Variable(tf.constant(0.2, shape=size))

def hyper_parameters_freq3(size):
    return tf.Variable(tf.constant(0.2, shape=size))

def hyper_parameters_amplitude(size):
    return tf.Variable(tf.constant(0.0, shape=size))
    
if __name__ == "__main__":

    # Domain bounds
    xmax = 0.12
    tmax = 7e-4
    x_c, y_c, u_c, v_c, p_c, rho_c, e_c, sat_c = preprocess('./Fluent-0.8.mat')
    x_c_2 = np.array([])
    y_c_2 = np.array([])
    u_c_2 = np.array([])
    v_c_2 = np.array([])
    p_c_2 = np.array([])
    rho_c_2 = np.array([])
    e_c_2 = np.array([])
    sat_c_2 = np.array([])
    

    
    for i in range(3901):
        if i%1 == 0:
            x_c_2 = np.append(x_c_2,x_c[i])
            y_c_2 = np.append(y_c_2,y_c[i])
            u_c_2 = np.append(u_c_2,u_c[i])
            v_c_2 = np.append(v_c_2,v_c[i])
            p_c_2 = np.append(p_c_2,p_c[i])
            rho_c_2 = np.append(rho_c_2,rho_c[i])
            e_c_2 = np.append(e_c_2,rho_c[i])
            sat_c_2 = np.append(sat_c_2,sat_c[i])
    x_c_2 = x_c_2.flatten()[:, None]
    y_c_2 = y_c_2.flatten()[:, None]
    u_c_2 = u_c_2.flatten()[:, None]
    v_c_2 = v_c_2.flatten()[:, None]
    p_c_2 = p_c_2.flatten()[:, None]
    rho_c_2 = rho_c_2.flatten()[:, None]
    e_c_2 = e_c_2.flatten()[:, None]
    sat_c_2 = sat_c_2.flatten()[:, None]

    x_exp = np.array([[0.060089],[0.064841],[0.069932],[0.074874],[0.080309],[0.082403],[0.083854],[0.086279],[0.088257],[0.089709],[0.092298],[0.094587],[0.097062],[0.099743],[0.104878],[0.109973],[0.114331],[0.119644]])
    y_exp = np.zeros_like(x_exp)
    #p_real = np.array([39184.77474],[36443.87887],[33511.24124],[30898.22036],[28016.43567],[27350.39649],[27272.73789],[27964.66327],[28737.92507],[28716.21178],[27800.08079],[26608.32543],[25451.08501],[24104.08318],[21768.24268],[19862.49501],[18060.99942],[16375.99172])
    p_real = np.array([[0.554026],[0.515273],[0.473809],[0.436864],[0.396119],[0.386702],[0.385604],[0.395387],[0.406320],[0.406013],[0.393060],[0.376210],[0.359848],[0.340803],[0.307777],[0.280832],[0.255361],[0.231537]])
    #e_c = e_c + (u_c**2+v_c**2)/2
    lb = np.array([0, 0, 0])
    ub = np.array([xmax, 0.02, tmax])
    
    # Network configuration
    uv_layers = [2] + 8*[50] + [5]
    uv_layers_sat = [2] + 5*[50] + [1]
    # Sample collocation points with Cartesian grid, another option is to sample with LHS, like in steady case
    a  = [hyper_parameters_A([1]) for l in range(1, len(uv_layers))]
    a1 = [hyper_parameters_amplitude([1]) for l in range(1, len(uv_layers))]
    a2 = [hyper_parameters_amplitude([1]) for l in range(1, len(uv_layers))]
    a3 = [hyper_parameters_amplitude([1]) for l in range(1, len(uv_layers))]
    a4 = [hyper_parameters_amplitude([1]) for l in range(1, len(uv_layers))]
    a5 = [hyper_parameters_amplitude([1]) for l in range(1, len(uv_layers))]
    a6 = [hyper_parameters_amplitude([1]) for l in range(1, len(uv_layers))]
    a7 = [hyper_parameters_amplitude([1]) for l in range(1, len(uv_layers))]
    a8 = [hyper_parameters_amplitude([1]) for l in range(1, len(uv_layers))]
    a9 = [hyper_parameters_amplitude([1]) for l in range(1, len(uv_layers))]
    
    F1 = [hyper_parameters_freq1([1]) for l in range(1, len(uv_layers))]
    F2 = [hyper_parameters_freq2([1]) for l in range(1, len(uv_layers))]
    F3 = [hyper_parameters_freq3([1]) for l in range(1, len(uv_layers))]
    F4 = [hyper_parameters_freq1([1]) for l in range(1, len(uv_layers))]
    F5 = [hyper_parameters_freq2([1]) for l in range(1, len(uv_layers))]
    F6 = [hyper_parameters_freq3([1]) for l in range(1, len(uv_layers))] 
    F7 = [hyper_parameters_freq1([1]) for l in range(1, len(uv_layers))]
    F8 = [hyper_parameters_freq2([1]) for l in range(1, len(uv_layers))]
    F9 = [hyper_parameters_freq3([1]) for l in range(1, len(uv_layers))] 

    x_c = x_c - 0.02
    x_c_2 = x_c_2 -0.02

    p_c_1 = (p_c_2-min(p_c_2))/(max(p_c_2)-min(p_c_2))


    u_c_1 = (u_c_2-min(u_c_2))/(max(u_c_2)-min(u_c_2))

    v_c_1 = (v_c_2-min(v_c_2))/(max(v_c_2)-min(v_c_2))


    
    e_c_1 = (e_c_2-min(e_c_2))/(max(e_c_2)-min(e_c_2))

    sat_c_1 = (sat_c_2-min(sat_c_2))/(max(sat_c_2)-min(sat_c_2))

    rho_c_1 = (rho_c_2-min(rho_c_2))/(max(rho_c_2)-min(rho_c_2))

    rho_l = 980
    
    p_sat = 2.8e4/(400**2 * 0.41567364)

    XY_c = np.concatenate((x_c_2, y_c_2),1)

    print(XY_c.shape)



    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        

        model_sat = SAT(XY_c, sat_c_1, uv_layers_sat, lb, ub)

        start_time = time.time()
        loss_sat = model_sat.train_sat(iter=5000, learning_rate=5e-4)
        model_sat.train_bfgs_sat()
        print("--- %s seconds ---" % (time.time() - start_time))

        # Print loss components
        model_sat.getloss_sat()
        model_sat.save_NN_sat('uvNN-stedy-sat-adaptive.pickle')
        sat = model_sat.predict_sat(XY_c[:,0:1],XY_c[:,1:2])
        sat = (sat*(max(sat_c)-min(sat_c))+min(sat_c))


        x_star = np.linspace(0, 0.12, 401)
        y_star = np.linspace(0, 0.02, 161)
        x_star, y_star = np.meshgrid(x_star, y_star)
        x_star = x_star.flatten()[:, None]
        y_star = y_star.flatten()[:, None]
        x_PINN_1 = np.array([])
        y_PINN_1 = np.array([])
        for i in range(len(x_star)):
            if  y_star[i]<-29188*pow(x_star[i],6) + 4728.7*pow(x_star[i],5) + 866.38*pow(x_star[i],4) - 263.45*pow(x_star[i],3) + 24.398*pow(x_star[i],2) - 0.9967*x_star[i] + 0.0204:
                x_PINN_1 = np.append(x_PINN_1,x_star[i])
                y_PINN_1 = np.append(y_PINN_1,y_star[i])
                
                x_PINN_1 = x_PINN_1.flatten()[:, None]
                y_PINN_1 = y_PINN_1.flatten()[:, None]  
        
        sat_pred = model_sat.predict_sat(x_PINN_1, y_PINN_1)
        # Train from scratch
        model = PINN_laminar_flow(XY_c, u_c_1, v_c_1, p_c_1, rho_c, e_c_1, sat, uv_layers, lb, ub)

        # Load trained model for inference
        #model = PINN_laminar_flow(XY_c, IC, INB, OUTB, WALL, u_c, v_c, p_c, rho_c, e_c, sat_c, uv_layers, lb, ub, ExistModel=1, uvDir='uvNN-unstedy-2.pickle')

        start_time = time.time()
        loss, loss_eq, loss_plt = model.train(iter=1000, learning_rate=5e-4)
        model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))

        # Print loss components
        model.getloss()

        # Save model for later use
        model.save_NN('uvNN-stedy-adaptive.pickle')
                
        
        #dst = ((x_star-0.2)**2+(y_star-0.2)**2)**0.5
        #x_star = x_star[dst >= 0.05]
        #y_star = y_star[dst >= 0.05]
            
        u_pred, v_pred, p_pred, e_pred = model.predict(x_PINN_1, y_PINN_1)
        field = [x_PINN_1, y_PINN_1, p_pred, sat_pred]
            #amp_pred = (u_pred**2 + v_pred**2)**0.5
            
        postProcess(xmin=0, xmax=0.12, ymin=0, ymax=0.02, field=field, s=2)
        
        
                    
        u_exp, v_exp, p_exp, e_exp = model.predict(x_exp, y_exp)
        p_c_2 = (p_exp*(max(p_c)-min(p_c))+min(p_c))/70727.321
        x = [x_exp, x_exp]
        y = [p_c_2,p_real]
        plt.scatter(x,y)
        
        x_ada = np.array([])
        y_ada = np.array([])
        for i in range(len(loss_plt)):
            if i%2==0:
                x_ada = np.append(x_ada, loss_plt[i])
            else:
                y_ada = np.append(y_ada, loss_plt[i])
        fig = plt.figure(dpi=300)
        plt.plot(x_ada[5:], y_ada[5:], c = 'blue')
        plt.plot(x_tra[5:], y_tra[5:], c = 'orange')
        plt.legend(['Adaptive activation function', 'Tanh activation function'])
