#coding: utf-8
import os
import numpy as np
import numpy.random as rd
import numpy.linalg as la
import scipy as sp
from scipy.stats import multivariate_normal
from collections import Counter

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc


def MixtureGaussian(data, mean, sigma, weight, D, cluster):
    PDF = np.zeros((N, cluster))
    for d in range(D):
        #混合正規分布
        #PDF[:, d] = [weight[d]*multivariate_normal.pdf(data_i, mean[d], sigma[d]) for data_i in data]
        for i, data_i in enumerate(data):
            PDF[i, d] = weight[d] * multivariate_normal.pdf(data_i, mean[d], sigma[d])
            #print("i: {0}, d: {1}, PDF: {2}".format(i, d, PDF[i, d]))
    return PDF


def gaussian(data, mean, sigma, weight, D, cluster):
    PDF = np.zeros((N, cluster))
    for d in range(D):
        for i, data_i in enumerate(data):

            #print("data_i: ", data_i)
            #print("mean[d]: ", mean[d])
            differ = data_i - mean[d]
            #print("hiku: ", differ)
            #print("sigma: ", sigma[d])

            #print("kake", np.dot(differ, np.linalg.inv(sigma[d])))
            
            product = np.dot(differ, np.linalg.inv(sigma[d]))

            numerator = np.exp( (-1) * 0.5 * np.dot(product, differ) )
            
            print("numerator: ", numerator)

            denominator = np.sqrt((2 * np.pi)**2 * la.det(sigma[d]))

            #print("denominator: ", 1/denominator )

            PDF[i, d] = (1 / denominator) * numerator
            #print("awase:", (1/denominator)*numerator)
            
            #print("i: {0}, d: {1}, PDF: {2}".format(i, d, PDF[i, d]))
    return PDF


# ======================================
# Parameters
# 混合数
cluster = 2
# 行数10000を指定
N = 10000
# 変数、データの次元数
D = 2
# パラメータの初期化
global weight
global mean
global sigma
# カラー
c = ['r', 'g', 'b']


#共分散
#sigma = np.abs(np.random.randn(cluster, D, D))
sigma = np.asanyarray(
    [ [[0.2,  0.0],[ 0.0, 0.2]],
        [[0.2,  0.0],[ 0.0, 0.2]] ])

print('sigma: ', sigma)
print('1分布に対する共分散sigma[d]', sigma[0].shape)


#重み
weight = np.zeros(cluster)
#制約条件(w[0]+w[1]=1)に従って各正規分布の初期重みを設定
for k in range(cluster):
    if k == cluster - 1:
        weight[k] = 1 - np.sum(weight)
    else:
        weight[k] = 1 / cluster
print('initial weight size: ', weight.shape)
print('initial weight: ', weight)


#ファイル読み込み, 2列10000行で取得
#data = np.loadtxt("data1000.txt", delimiter=',', unpack=True)
data = np.loadtxt("data1000.txt", delimiter=',')
#print(data.shape)
#(10000, 2)
#print(data)

#平均
max_x, min_x = np.max(data[:,0]), np.min(data[:,0])
max_y, min_y = np.max(data[:,1]), np.min(data[:,1])
mean = np.c_[rd.uniform(low=min_x, high=max_x, size=cluster), rd.uniform(low=min_y, high=max_y, size=cluster) ]
print('mean shape: ', mean.shape)
print('mean: ', mean)

'''
trans_data = data.transpose()
plt.scatter(trans_data[0, :], trans_data[1, :], c='gray', alpha=0.5, marker="+")
plt.show()
'''

print('===========================================')

# ======================================
#イテレーション100回
for iteration in range(3):
    
    print('iteration:', iteration)

    # E ステップ ========================================================================
    # 対数尤度の期待値計算
    # 2次元の混合正規分布を形成
    MG = MixtureGaussian(data, mean, sigma, weight, D, cluster)
    #MG2 = gaussian(data, mean, sigma, weight, D, cluster)
    
    #print(MG.shape)
    print(MG)
    # モデルパラメータ値で計算される条件付き確率を算出
    MG_sum = np.sum(MG, axis=1)
    print("MG_sum: ", MG_sum.shape)
    print("MG_sum: ", MG_sum)
    MGT = MG.T
    print("MG.T: ", MGT.shape)
    CP = (MG.T/MG_sum).T

    '''
    # M ステップ ========================================================================
    # data[i]をすべて合計し，dataがk番目の分布から発生されたと考える確率を算出
    CP_sum = [np.sum(CP[:,d]) for d in range(D)]
    print(CP_sum)

    # 重みの更新
    weight =  CP_sum/N
    '''

    '''
    # calculate mean
    tmp_mu = np.zeros((D, D))

    for d in range(D):
        for i in range(len(data)):
            tmp_mu[d] += CP[i, d]*data[i]
        tmp_mu[d] = tmp_mu[d]/CP_sum[d]
    mu_prev = mean.copy()
    mean = tmp_mu.copy()

    # calculate sigma
    tmp_sigma = np.zeros((D, D, D))

    for d in range(D):
        tmp_sigma[d] = np.zeros((D, D))
        for i in range(N):
            tmp = np.asanyarray(data[i]-mean[d])[:,np.newaxis]
            tmp_sigma[d] += CP[i, d]*np.dot(tmp, tmp.T)
        tmp_sigma[d] = tmp_sigma[d/CP_sum[d]

    sigma = tmp_sigma.copy()

    # calculate likelihood
    prev_likelihood = MG
    MG = MixtureGaussian(data, mean, sigma, weight, D, cluster)

    prev_sum_log_likelihood = np.sum(np.log(prev_likelihood))
    sum_log_likelihood = np.sum(np.log(MG))
    diff = prev_sum_log_likelihood - sum_log_likelihood

    print('sum of log likelihood:', sum_log_likelihood)
    print('diff:', diff)

    print('weight:', weight)
    print('mean:', mean)
    print('sigma:', sigma)

    # visualize
    for i in range(N):
        plt.scatter(data[i,0], data[i,1], s=30, c=CP[i], alpha=0.5, marker="+")

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
    '''