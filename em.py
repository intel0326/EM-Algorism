#coding: utf-8
import os
import numpy as np
import numpy.random as rd
import scipy as sp
from scipy import stats as st
from collections import Counter

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc


def calc_likelihood(data, mean, sigma, weight, D):
    likelihood = np.zeros((N, 2))
    for d in range(D):
        likelihood[:, d] = [weight[d]*st.multivariate_normal.pdf(i, mean[d], sigma[d]) for i in data]
    return likelihood


# ======================================
# 山下先生のParameters
cluster = 2
#行数10000を指定
N = 10000
#変数、データの次元数
D = 2

#ファイル読み込み, 2列10000行で取得
data = np.loadtxt("data1000.txt", delimiter=',', unpack=True)

'''
print(data.shape)
print(data)
print(type(data))

plt.scatter(data[0,:], data[1,:], c='gray', alpha=0.5, marker="+")
plt.show()
'''

# ======================================
#イテレーション100回
for iteration in range(100):
    
    #パラメータの初期化
    global weight
    global mean
    global sigma
    print('iteration:', iteration)


    # E step ========================================================================
    # calculate responsibility(負担率)
    likelihood = calc_likelihood(data, mean, sigma, weight, D)
    gamma = (likelihood.T/np.sum(likelihood, axis=1)).T
    N_k = [np.sum(gamma[:,d]) for d in range(D)]

    # M step ========================================================================

    # caluculate weight
    weight =  N_k/N

    # calculate mean
    tmp_mu = np.zeros((D, D))

    for d in range(D):
        for i in range(len(data)):
            tmp_mu[d] += gamma[i, d]*data[i]
        tmp_mu[d] = tmp_mu[d]/N_k[d]
    mu_prev = mean.copy()
    mean = tmp_mu.copy()

    # calculate sigma
    tmp_sigma = np.zeros((D, D, D))

    for d in range(D):
        tmp_sigma[d] = np.zeros((D, D))
        for i in range(N):
            tmp = np.asanyarray(data[i]-mean[d])[:,np.newaxis]
            tmp_sigma[d] += gamma[i, d]*np.dot(tmp, tmp.T)
        tmp_sigma[d] = tmp_sigma[d]/N_k[d]

    sigma = tmp_sigma.copy()

    # calculate likelihood
    prev_likelihood = likelihood
    likelihood = calc_likelihood(data, mean, sigma, weight, D)

    prev_sum_log_likelihood = np.sum(np.log(prev_likelihood))
    sum_log_likelihood = np.sum(np.log(likelihood))
    diff = prev_sum_log_likelihood - sum_log_likelihood

    print('sum of log likelihood:', sum_log_likelihood)
    print('diff:', diff)

    print('weight:', weight)
    print('mean:', mean)
    print('sigma:', sigma)

    # visualize
    for i in range(N):
        plt.scatter(data[i,0], data[i,1], s=30, c=gamma[i], alpha=0.5, marker="+")

    for i in range(D):
        ax = plt.axes()
        ax.arrow(mu_prev[i, 0], mu_prev[i, 1], mean[i, 0]-mu_prev[i, 0], mean[i, 1]-mu_prev[i, 1],
                  lw=0.8, head_width=0.02, head_length=0.02, fc='k', ec='k')
        plt.scatter([mu_prev[i, 0]], [mu_prev[i, 1]], c=c[i], marker='o', alpha=0.8)
        plt.scatter([mean[i, 0]], [mean[i, 1]], c=c[i], marker='o', edgecolors='k', linewidths=1)
    plt.title("step:{}".format(iteration))

    #print_gmm_contour(mean, sigma, weight, D)

    if np.abs(diff) < 0.0001:
        plt.title('likelihood is converged.')
    else:
        plt.title("iter:{}".format(iteration-3))