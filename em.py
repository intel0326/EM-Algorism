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
        #print("sigma: ", sigma[d])
        for i, data_i in enumerate(data):
            PDF[i, d] = weight[d] * multivariate_normal.pdf(data_i, mean[d], sigma[d])
            #print("i: {0}, d: {1}, PDF: {2}".format(i, d, PDF[i, d]))
    return PDF


def gaussian(data, mean, sigma, weight, D, cluster):
    '''
    PDF = np.zeros((N, cluster))
    for d in range(D):
        for i, data_i in enumerate(data):

            #print("data_i: ", data_i)
            #print("mean[d]: ", mean[d])
            differ = data_i - mean[d]
            print("hiku: ", differ)
            #print("sigma: ", sigma[d])
            
            product = np.dot(differ, np.linalg.inv(sigma[d]))
            print("inv: ", np.linalg.inv(sigma[d]))
            print("product: ", product)

            fact = (-1) * 0.5 * np.dot(product, differ)
            print('fact: ', fact)

            numerator = np.exp( fact )
            
            print("numerator: ", numerator)

            denominator = np.sqrt((2 * np.pi)**2 * la.det(sigma[d]))

            #print("denominator: ", 1/denominator )

            PDF[i, d] = (1 / denominator) * numerator
            #print("awase:", (1/denominator)*numerator)
            
            print("i: {0}, d: {1}, PDF: {2}".format(i, d, PDF[i, d]))
    '''
    PDF = np.zeros((N, cluster))
    for k in range(cluster):
        for i, data_i in enumerate(data):

            #print("mean[k]: ", mean[k])
            differ = data_i - mean[k]
            #print("hiku: ", differ)
            #print("sigma: ", sigma[d])
            
            product = np.dot(differ, np.linalg.inv(sigma[k]))
            #print("inv: ", np.linalg.inv(sigma[k]))
            #print("product: ", product)

            fact = (-1) * 0.5 * np.dot(product, differ)
            #print('fact: ', fact)

            numerator = np.exp( fact )
            
            #print("numerator: ", numerator)

            denominator = np.sqrt((2 * np.pi)**2 * la.det(sigma[k]))

            #print("denominator: ", 1/denominator )

            PDF[i, k] = weight[k] * (1 / denominator) * numerator
            #print("awase:", (1/denominator)*numerator)
            
            #print("i: {0}, d: {1}, PDF: {2}".format(i, k, PDF[i, k]))
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
#mean = np.asanyarray( [ [103.84,  1140.86],[110.294,  1089.27] ] )
print('mean shape: ', mean.shape)
print('mean: ', mean)
#x=103.84 y=1140.86
#x=110.294 y=1089.27


#共分散
'''
sigma = np.asanyarray(
    [ [[100,  0.0],[ 0.0, 100]],
        [[100,  0.0],[ 0.0, 100]] ])
''' 
#各要素2乗
rho = np.power(data,2)
#分散を求めるため，列ごとに足す
rho = np.sum(rho, axis=0)
rho_x = rho[0] / 10000
rho_y = rho[1] / 10000
print("rho :", rho)
sigma = np.asanyarray(
    [ [[rho_x,  0.0],[ 0.0, rho_y]],
        [[rho_x,  0.0],[ 0.0, rho_y]] ])
print('sigma: ', sigma)
print('1分布に対する共分散sigma[d]', sigma[0].shape)



trans_data = data.transpose()
plt.scatter(trans_data[0, :], trans_data[1, :], c='gray', alpha=0.5, marker="+")
plt.show()


print('===========================================')

# ======================================
#イテレーション100回
for iteration in range(100):


    print('iteration:', iteration)


    # E ステップ ========================================================================

    # 対数尤度の期待値計算
    # 2次元の混合正規分布を形成
    MG = MixtureGaussian(data, mean, sigma, weight, D, cluster)
    MG2 = gaussian(data, mean, sigma, weight, D, cluster)
    #print("MG: ", MG)
    #print(MG.shape)
    #print("MG2: ", MG2)


    # モデルパラメータ値で計算される条件付き確率を算出
    MG_sum = np.sum(MG, axis=1)
    #print("MG_sum: ", MG_sum.shape)
    #print("MG_sum: ", MG_sum)
    #負担率CPを求める
    CP = (MG.T/MG_sum).T
    #print("CP: ", CP)

    

    # M ステップ ========================================================================
    # data[d]をすべて合計し，dataがk番目の分布から発生されたと考える確率を算出
    CP_sum = [np.sum(CP[:,d]) for d in range(D)]
    CP_sum = np.array(CP_sum)
    #print("CP_sum: ", CP_sum)



    # 重みの更新
    weight =  CP_sum / N
    #print("weight: ", weight)
    


    # 平均の更新
    mean_temp = np.zeros((D, D))

    #要編集
    for d in range(D):
        for i in range(len(data)):
            mean_temp[d] += CP[i, d]*data[i]
        mean_temp[d] = mean_temp[d]/CP_sum[d]
    mean_pre = mean.copy()
    mean = mean_temp.copy()



    # 共分散行列の更新
    sigma_temp = np.zeros((cluster, D, D))

    #要編集
    for d in range(D):
        sigma_temp[d] = np.zeros((D, D))
        for i in range(N):
            temp = np.asanyarray(data[i]-mean[d])[:,np.newaxis]
            sigma_temp[d] += CP[i, d]*np.dot(temp, temp.T)
        sigma_temp[d] = sigma_temp[d] / CP_sum[d]

    sigma = sigma_temp.copy()



    # calculate likelihood
    '''
    prev_likelihood = MG
    MG = MixtureGaussian(data, mean, sigma, weight, D, cluster)

    prev_sum_log_likelihood = np.sum(np.log(prev_likelihood))
    sum_log_likelihood = np.sum(np.log(MG))
    diff = prev_sum_log_likelihood - sum_log_likelihood

    print('sum of log likelihood:', sum_log_likelihood)
    print('diff:', diff)
    '''

    print('weight:', weight)
    print('mean:', mean)
    print('sigma:', sigma)

    # visualize
    '''
    for i in range(N):
        plt.scatter(data[i,0], data[i,1], s=30, c=CP[i], alpha=0.5, marker="+")

    for i in range(D):
        ax = plt.axes()
        ax.arrow(mu_pre[i, 0], mu_pre[i, 1], mean[i, 0]-mu_pre[i, 0], mean[i, 1]-mu_pre[i, 1],
                  lw=0.8, head_width=0.02, head_length=0.02, fc='k', ec='k')
        plt.scatter([mu_pre[i, 0]], [mu_pre[i, 1]], c=c[i], marker='o', alpha=0.8)
        plt.scatter([mean[i, 0]], [mean[i, 1]], c=c[i], marker='o', edgecolors='k', linewidths=1)
    plt.title("step:{}".format(iteration))

    #print_gmm_contour(mean, sigma, weight, D)
    '''


    '''
    if np.abs(diff) < 0.0001:
        plt.title('likelihood is converged.')
    else:
        plt.title("iter:{}".format(iteration-3))
    '''