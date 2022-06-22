import sys,codecs, time, math, argparse, ot
import numpy as np
from utils import *

parser = argparse.ArgumentParser(description='Wasserstein Procrustes for Embedding Alignment')
parser.add_argument('--model_src', default='',type=str, help='Path to source word embeddings')
parser.add_argument('--model_tgt', default='',type=str, help='Path to target word embeddings')
parser.add_argument('--lexicon',default='', type=str, help='Path to the evaluation lexicon')
parser.add_argument('--output_src', default='', type=str, help='Path to save the aligned source embeddings')
parser.add_argument('--output_tgt', default='', type=str, help='Path to save the aligned target embeddings')

parser.add_argument('--seed', default=1111, type=int, help='Random number generator seed')
parser.add_argument('--nepoch', default=5, type=int, help='Number of epochs')
parser.add_argument('--niter', default=5000, type=int, help='Initial number of iterations')
parser.add_argument('--bsz', default=500, type=int, help='Initial batch size')
parser.add_argument('--lr', default=500., type=float, help='Learning rate')
parser.add_argument('--nmax', default=20000, type=int, help='Vocabulary size for learning the alignment')
parser.add_argument('--reg', default=0.05, type=float, help='Regularization parameter for sinkhorn')
args = parser.parse_args()

def objective(X, Y, R, n=5000):
    Xn, Yn = X[:n], Y[:n]  #[5000,300]  [5000,300]
    C = -np.dot(np.dot(Xn, R), Yn.T)  #[5000,5000]
    P = ot.sinkhorn(np.ones(n), np.ones(n), C, 0.025, stopThr=1e-3)   #优化 [5000,5000]
    return 1000 * np.linalg.norm(np.dot(Xn, R) - np.dot(P, Yn)) / n  #是个值

def sqrt_eig(x):  #2500*300
    U, s, VT = np.linalg.svd(x, full_matrices=False)   #奇异值分解 U[2500,300]  s[300,300]  VT[300,300] s是除对角线外其他值都为0的对角矩阵
    return np.dot(U, np.dot(np.diag(np.sqrt(s)), VT))   #[2500,300][1,300][300,300]  重新还原后的矩阵与原来相似

#Wasserstein Procrustes
def align(X, Y, R, lr=10., bsz=200, nepoch=5, niter=1000,
          nmax=10000, reg=0.05, verbose=True):  #1个iteration等于使用batchsize个样本训练一次；1个epoch等于使用训练集中的全部样本训练一次
    for epoch in range(1, nepoch + 1):  #5次
        for _it in range(1, niter + 1):  #5000次
            # sample mini-batch
            xt = X[np.random.permutation(nmax)[:bsz], :]   #在X中任意取500行[500,300]
            yt = Y[np.random.permutation(nmax)[:bsz], :]  #在Y中任意取500行[500,300]
            # compute OT on minibatch
            C = -np.dot(np.dot(xt, R), yt.T)  #[500,500]
            P = ot.sinkhorn(np.ones(bsz), np.ones(bsz), C, reg, stopThr=1e-3)  #优化
            # compute gradient
            G = - np.dot(xt.T, np.dot(P, yt))  #[300,300]
            R -= lr / bsz * G  #梯度下降更新R
            # project on orthogonal matrices
            U, s, VT = np.linalg.svd(R)  #奇异值分解
            R = np.dot(U, VT)  #新的R  [300,300]
        bsz *= 2
        niter //= 4  #整除运算
        if verbose:
            print("epoch: %d  obj: %.3f" % (epoch, objective(X, Y, R)))
    return R    #输出优化好后的R  [300,300]

#convex relaxation 凸优化
def convex_init(X, Y, niter=100, reg=0.05, apply_sqrt=False):  #源嵌入向量前2500行，目标嵌入向量前2500行，迭代次数100，正则化参数0.05，True
    n, d = X.shape  #n=2500,d=300
    if apply_sqrt: #奇异值分解 分解后还是[2500,300]和[2500,300]
        X, Y = sqrt_eig(X), sqrt_eig(Y)
    K_X, K_Y = np.dot(X, X.T), np.dot(Y, Y.T)  #[2500,2500]  [2500,2500]
    K_Y *= np.linalg.norm(K_X) / np.linalg.norm(K_Y)   #[2500,2500]  np.linalg.norm(K_X)是个值
    K2_X, K2_Y = np.dot(K_X, K_X), np.dot(K_Y, K_Y)  #[2500,2500]  [2500,2500]
    P = np.ones([n, n]) / float(n)  #[2500,2500]
    for it in range(1, niter + 1):
        G = np.dot(P, K2_X) + np.dot(K2_Y, P) - 2 * np.dot(K_Y, np.dot(P, K_X))   #目标损失
        q = ot.sinkhorn(np.ones(n), np.ones(n), G, reg, stopThr=1e-3)  #优化
        alpha = 2.0 / float(2.0 + it)
        P = alpha * q + (1.0 - alpha) * P  #更新P  一共更新99轮P值
    obj = np.linalg.norm(np.dot(P, K_X) - np.dot(K_Y, P))
    print(obj)
    return procrustes(np.dot(P, X), Y).T

print("\n*** Wasserstein Procrustes ***\n")
# np.random.seed(args.seed)  #1111
maxload = 200000
w_src, x_src = load_vectors(args.model_src, maxload, norm=True, center=True) #Path to source word embeddings返回词语列表 以及 预处理后的矩阵[20万,300]
w_tgt, x_tgt = load_vectors(args.model_tgt, maxload, norm=True, center=True) #Path to target word embeddings返回词语列表 以及 预处理后的矩阵[20万,300]
src2trg, _ = load_lexicon(args.lexicon, w_src, w_tgt)  #Path to the evaluation lexicon将词典信息和源、目标嵌入信息连接起来 字典类型，百分比

print("\nComputing initial mapping with convex relaxation...")  #用凸松弛法计算初始映射...  RCSCL
t0 = time.time()
R0 = convex_init(x_src[:2500], x_tgt[:2500], reg=args.reg, apply_sqrt=True)  #只要源嵌入矩阵的前2500行，目标嵌入的前2500行 sinkhorn正则化参数为0.05
print("Done [%03d sec]" % math.floor(time.time() - t0))   #R0为[300,300]

print("\nComputing mapping with Wasserstein Procrustes...")  #用瓦瑟斯坦·普罗科斯特斯计算映射
t0 = time.time()
R = align(x_src, x_tgt, R0.copy(), bsz=args.bsz, lr=args.lr, niter=args.niter,
          nepoch=args.nepoch, reg=args.reg, nmax=args.nmax)   #源嵌入向量 目标嵌入向量 凸优化的结果 每次输入=500 学习率=500 迭代次数=5000
print("Done [%03d sec]" % math.floor(time.time() - t0))  #epochs=5 正则化参数=0.05 词汇量大小=20000  输出优化好的R[300,300]

acc1 = compute_nn_accuracy(x_src, np.dot(x_tgt, R.T), src2trg)   #选出top-one的准确率
print("\nPrecision@1: %.3f\n" % acc1)

acc2 = compute_csls_accuracy(x_src, np.dot(x_tgt, R.T), src2trg)   #选出top-k的去中心化，即弱化Hubness现象
print("\nPrecision@1: %.3f\n" % acc2)

if args.output_src != '':  #源嵌入重新写回
    x_src = x_src / np.linalg.norm(x_src, 2, 1).reshape([-1, 1])  #求行向量的范数[20万,300]
    save_vectors(args.output_src, x_src, w_src)  #Path to save the aligned source embeddings
if args.output_tgt != '':   #目标嵌入乘上W写回 即将目标向量乘上W映射到源空间
    x_tgt = x_tgt / np.linalg.norm(x_tgt, 2, 1).reshape([-1, 1])  #求行向量的范数[20万,300]
    save_vectors(args.output_tgt, np.dot(x_tgt, R.T), w_tgt)   #Path to save the aligned target embeddings