#coding: utf-8
import os
import os.path as path
import numpy as np
import numpy.random as rd
import numpy.linalg as la
import scipy as sp
from scipy.stats import multivariate_normal
from collections import Counter

import glob
from PIL import Image

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import matplotlib.animation as animation

def first_plot(data, mean):

    if path.exists("result") == False:
        os.mkdir('result')

    plt.figure()
    trans_data = data.transpose()
    plt.scatter(trans_data[0, :], trans_data[1, :], c='gray', alpha=0.5, marker="+")
    plt.scatter([mean[0, 0]], [mean[0, 1]], c=c[0], marker='o', edgecolors='k', linewidths=1)
    plt.scatter([mean[1, 0]], [mean[1, 1]], c=c[1], marker='o', edgecolors='k', linewidths=1)
    plt.savefig("result/EM_first.png", dpi=300)


def plot(data, D, mean, mean_pre, c, iteration):

    plt.figure()
    trans_data = data.transpose()
    plt.scatter(trans_data[0, :], trans_data[1, :], c='gray', alpha=0.5, marker="+")

    #要編集
    for i in range(D):
        ax = plt.axes()
        ax.arrow(mean_pre[i, 0], mean_pre[i, 1], mean[i, 0]-mean_pre[i, 0], mean[i, 1]-mean_pre[i, 1],
                  lw=0.8, head_width=0.02, head_length=0.02, fc='k', ec='k')
        plt.scatter([mean_pre[i, 0]], [mean_pre[i, 1]], c=c[i], marker='o', edgecolors='k', alpha=0.8, linewidths=1)
        plt.scatter([mean[i, 0]], [mean[i, 1]], c=c[i], marker='o', edgecolors='k', linewidths=1)
    plt.title("step:{}".format(iteration))
    #plt.show()
    plt.savefig("result/EM_result_{0}.png".format(iteration), dpi=300)


def gaussian(data, mean, sigma, weight, D, cluster):
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
            
    return PDF


# ======================================
# パラメータ

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


# ======================================
# 重み
weight = np.zeros(cluster)

# 制約条件(w[0]+w[1]=1)に従って各正規分布の初期重みを設定
for k in range(cluster):
    if k == cluster - 1:
        weight[k] = 1 - np.sum(weight)
    else:
        weight[k] = 1 / cluster
print('initial weight size: ', weight.shape)
print('initial weight: ', weight)


# ======================================
# ファイル読み込み
data = np.loadtxt("data1000.txt", delimiter=',')
#data = np.loadtxt("data1000.txt", delimiter=',', unpack=True)


# ======================================
# 平均
# dataの要素の最小値，最大値を取得し，その中でランダムな値を初期値にする
max_x, min_x = np.max(data[:,0]), np.min(data[:,0])
max_y, min_y = np.max(data[:,1]), np.min(data[:,1])
mean = np.c_[rd.uniform(low=min_x, high=max_x, size=cluster), rd.uniform(low=min_y, high=max_y, size=cluster) ]
#mean = np.asanyarray( [ [103.84,  1140.86],[110.294,  1089.27] ] )
print('mean shape: ', mean.shape)
print('mean: ', mean)


# ======================================
# 共分散
# 各要素の2乗
rho = np.power(data,2)
# 分散を求めるため，列ごとに足す
rho = np.sum(rho, axis=0)
print("rho :", rho)
# 次元ごとに除算，今回は2次元(x座標，y座標)であるため，除算結果を2変数に格納
rho_x = rho[0] / 10000
rho_y = rho[1] / 10000
# 対角共分散行列を作製
sigma = np.asanyarray(
    [ [[rho_x,  0.0],[ 0.0, rho_y]],
        [[rho_x,  0.0],[ 0.0, rho_y]] ])
print('sigma: ', sigma)
print('1分布に対する共分散sigma[d]', sigma[0].shape)



# ======================================
# ビジュアライズ
# 出力先の確認と作製
first_plot(data, mean)


# ======================================
#イテレーション回数は任意
for iteration in range(100):

    print('===========================================')
    print('iteration:', iteration)


    # E ステップ ========================================================================

    # 対数尤度の期待値計算
    # 2次元の混合正規分布を形成
    MG = gaussian(data, mean, sigma, weight, D, cluster)
    #print("MG: ", MG)
    #print(MG.shape)


    # ======================================
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


    # ======================================
    # 重みの更新
    weight =  CP_sum / N
    #print("weight: ", weight)
    

    # ======================================
    # 平均の更新
    mean_temp = np.zeros((D, D))

    #要編集
    for d in range(D):
        for i in range(len(data)):
            mean_temp[d] += CP[i, d]*data[i]
        mean_temp[d] = mean_temp[d]/CP_sum[d]
    mean_pre = np.copy(mean)
    mean = mean_temp.copy()


    # ======================================
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


    # ======================================
    # 各種パラメータの出力
    print('weight:', weight)
    print('mean:', mean)
    print('sigma:', sigma)


    # ======================================
    # ビジュアライズ
    #trans_data = data.transpose()
    #plt.scatter(trans_data[0, :], trans_data[1, :], c='gray', alpha=0.5, marker="+")
    plot(data, D, mean, mean_pre, c, iteration)


    # ======================================
    # 収束条件の計算と，プログラム終了判定
    MG_pre = MG
    MG = gaussian(data, mean, sigma, weight, D, cluster)
    
    #要編集
    sum_log_MG_pre = np.sum(np.log(MG_pre))
    sum_log_MG = np.sum(np.log(MG))
    diff = sum_log_MG_pre - sum_log_MG
    print('sum of log MG:', sum_log_MG)
    print('diff:', diff)

    if np.abs(diff) < 0.0001:
        #plt.title('convergence.')
        print('収束しました')
        break

