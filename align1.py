#将汉语嵌入到西语空间中
#有监督的方法
import numpy as np
import argparse
from fastText.alignment.utils import *
import sys

parser = argparse.ArgumentParser(description='RCSLS for supervised word alignment')
parser.add_argument("--src_emb", type=str, default=r'g:\my paper\单语嵌入\cc.zh.300.vec', help="Load source embeddings")  #####
parser.add_argument("--tgt_emb", type=str, default=r'g:\my paper\单语嵌入\cc.es.300.vec', help="Load target embeddings")   #####
parser.add_argument('--center', action='store_true', help='whether to center embeddings or not') #只要运行时该变量有传参就将该变量设为True
parser.add_argument("--dico_train", type=str, default=r'g:\my paper\字典\zh-es_train.txt', help="train dictionary")  #####
parser.add_argument("--dico_test", type=str, default=r'g:\my paper\字典\zh-es_validtion.txt', help="validation dictionary")   #####
parser.add_argument("--output", type=str, default=r'g:\my paper\双语嵌入结果\zh_es_embeding.txt', help="where to save aligned embeddings")  #####

parser.add_argument("--knn", type=int, default=10, help="number of nearest neighbors in RCSL/CSLS")
parser.add_argument("--maxneg", type=int, default=200000, help="Maximum number of negatives for the Extended RCSLS")
parser.add_argument("--maxsup", type=int, default=-1, help="Maximum number of training examples")
parser.add_argument("--maxload", type=int, default=200000, help="Maximum number of loaded vectors")
parser.add_argument("--model", type=str, default="none", help="Set of constraints: spectral or none")
parser.add_argument("--reg", type=float, default=0.0 , help='regularization parameters')
parser.add_argument("--lr", type=float, default=1.0, help='learning rate')
parser.add_argument("--niter", type=int, default=10, help='number of iterations')
parser.add_argument('--sgd', action='store_true', help='use sgd')  #只要运行时该变量有传参就将该变量设为True
parser.add_argument("--batchsize", type=int, default=10000, help="batch size for sgd")
params = parser.parse_args()

###### SPECIFIC FUNCTIONS ######

def getknn(sc, x, y, k=10):
    sidx = np.argpartition(sc, -k, axis=1)[:, -k:]  #为随机选出的1万个x的映射后的x在20万个目标中选择最近的10个 输出的矩阵为索引矩阵  [1万,10]
    ytopk = y[sidx.flatten(), :]  #在负样本Y中找到这10万个索引的对应向量 [10万，300]
    ytopk = ytopk.reshape(sidx.shape[0], sidx.shape[1], y.shape[1])  #[1万,10,300]  变换成1万个[10,300]的矩阵  每10个是一个词的top-10
    f = np.sum(sc[np.arange(sc.shape[0])[:, None], sidx])  #在sc的第0行取出sidx第0行的10个索引所对应的值（相似值） 以此类推
    df = np.dot(ytopk.sum(1).T, x)  #上面一行的意思是：为1万个映射后的x中的每一个找到10个最佳的相似度值加和 成为一个值f
    return f / k, df / k   #df[300,300] ytopk.sum(1)为1万个中的每一个矩阵列值相加 共[1万,300]  返回一个值和一个[300,300]的矩阵

def rcsls(X_src, Y_tgt, Z_src, Z_tgt, R, knn=10):
    X_trans = np.dot(X_src, R.T)  #给X映射到目标空间  [1万,300]
    f = 2 * np.sum(X_trans * Y_tgt)  #值
    df = 2 * np.dot(Y_tgt.T, X_src)  #[300,300]
    fk0, dfk0 = getknn(np.dot(X_trans, Z_tgt.T), X_src, Z_tgt, knn)  ##返回损失值的第二部分，及第二部分的导数  （从Y中获取X最大的10个）
    fk1, dfk1 = getknn(np.dot(np.dot(Z_src, R.T), Y_tgt.T).T, Y_tgt, Z_src, knn) ##返回损失值的第三部分，及第三部分的导数  （从X中获取Y最大的10个）
    f = f - fk0 -fk1  ##损失值
    df = df - dfk0 - dfk1.T  ##损失值的导数，用于以梯度下降的方式找到最佳的映射矩阵W
    return -f / X_src.shape[0], -df / X_src.shape[0]

def proj_spectral(R):
    U, s, V = np.linalg.svd(R)
    s[s > 1] = 1
    s[s < 0] = 0
    return np.dot(U, np.dot(np.diag(s), V))

# load word embeddings
words_src, x_src = load_vectors(params.src_emb, maxload=params.maxload, center=params.center)  #输出源嵌入中的词语列表及[20万,300]的矩阵
words_tgt, x_tgt = load_vectors(params.tgt_emb, maxload=params.maxload, center=params.center)  #输出目标嵌入中的词语列表 长度为20万；以及[20万,300]的矩阵

# load validation bilingual lexicon
src2tgt, lexicon_size = load_lexicon(params.dico_test, words_src, words_tgt)#返回字典类型的双语词典对应的在源目标嵌入中的索引；双语词典的长度

# word --> vector indices
idx_src = idx(words_src) #源嵌入词语对应的在源嵌入中的索引号 一一对应的因为嵌入中不存在重复现象 {词:索引,词:索引}
idx_tgt = idx(words_tgt)  #目标嵌入词语对应的在目标嵌入中的索引号 一一对应的 {词:索引,词:索引}

# load train bilingual lexicon
pairs = load_pairs(params.dico_train, idx_src, idx_tgt)  #加载训练词典 返回训练双语词典中的一对词在源嵌入和目标嵌入中的索引[(1,2),(5,7)]
if params.maxsup > 0 and params.maxsup < len(pairs):   #-1>0 and -1<训练双语词典词对在源、目标嵌入中的数量  （按理来说应该比原始训练双语词典的要小一点）
    pairs = pairs[:params.maxsup]  #用来限制训练样本的大小 但是没用到！

# selecting training vector pairs
X_src, Y_tgt = select_vectors_from_pairs(x_src, x_tgt, pairs)  #得到的结果是一一对应的，返回训练双语词典中的源词所对应的在源嵌入向量中的向量 [训练长度,300]  [训练长度,300]

# adding negatives for RCSLS
Z_src = x_src[:params.maxneg, :]  #取源嵌入中的20万个向量作为负样本  [20万,300]
Z_tgt = x_tgt[:params.maxneg, :]  #取目标嵌入中的20万个向量作为负样本  [20万,300]

# initialization:
R = procrustes(X_src, Y_tgt)  #训练向量乘积的奇异值分解  [300,300]  相当于没有优化的w值
nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=lexicon_size)  #top-1
print("[init -- Procrustes] NN: %.4f"%(nnacc))
sys.stdout.flush()  #不能缓满了之后才输出，而是及时输出关于准确率的信息
cslsacc = compute_csls_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=lexicon_size)  #余弦相似度减去top-k均值后取top-1
print("[init -- Procrustes] CSLS: %.4f"%(cslsacc))
sys.stdout.flush()  #不能缓满了之后才输出，而是及时输出关于准确率的信息

# optimization
fold, Rold = 0, []
niter, lr = params.niter, params.lr   #10 & 1.0
for it in range(0, niter + 1): #11次
    if lr < 1e-4:
        break
    if params.sgd:  #未激发
        indices = np.random.choice(X_src.shape[0], size=params.batchsize, replace=False)  #从训练长度中随机选择1万个值
        f, df = rcsls(X_src[indices, :], Y_tgt[indices, :], Z_src, Z_tgt, R, params.knn) #返回一个值和一个[1万,300]的矩阵
    else:
        f, df = rcsls(X_src, Y_tgt, Z_src, Z_tgt, R, params.knn)  #返回一个值和一个[300,300]的矩阵  返回损失的值，及损失对W的导数
    if params.reg > 0: #未激发
        R *= (1 - lr * params.reg)
    R -= lr * df  ##梯度下降来优化W值
    if params.model == "spectral":   #Set of constraints: spectral or none  默认不加spectral约束
        R = proj_spectral(R)
    print("[it=%d] f = %.4f" % (it, f))  #打印迭代次数以及当前的损失值
    sys.stdout.flush()
    if f > fold and it > 0 and not params.sgd:   #第一次迭代不执行  从第二次迭代开始执行 新f比之前的f大则用之前的值覆盖他们
        lr /= 2   #学习率减半
        f, R = fold, Rold  ##损失值f越小越好，如果f比之前的大，那就把之前的值赋给f，之前的W赋给W
    fold, Rold = f, R  #f<=fold 或者 第二次迭代 或者执行了sgd  新f比之前的小则用新的覆盖之前的f
    if (it > 0 and it % 10 == 0) or it == niter:   #第10次迭代的时候计算准确度
        nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=lexicon_size)
        print("[it=%d] NN = %.4f - Coverage = %.4f" % (it, nnacc, len(src2tgt) / lexicon_size))
        cslsacc=compute_csls_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=lexicon_size)
        print("[it=%d] CSLS = %.4f - Coverage = %.4f" % (it, cslsacc, len(src2tgt) / lexicon_size))
nnacc = compute_nn_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=lexicon_size)
print("[final] NN = %.4f - Coverage = %.4f" % (nnacc, len(src2tgt) / lexicon_size))
cslsacc=compute_csls_accuracy(np.dot(x_src, R.T), x_tgt, src2tgt, lexicon_size=lexicon_size)
print("[final] CSLS = %.4f - Coverage = %.4f" % (cslsacc, len(src2tgt) / lexicon_size))

if params.output != "":
    print("Saving all aligned vectors at %s" % params.output)
    words_full, x_full = load_vectors(params.src_emb, maxload=-1, center=params.center, verbose=False) #源嵌入的200万词列表；源嵌入向量[200万,300]
    x = np.dot(x_full, R.T)  #映射到目标空间
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8  #范式
    save_vectors(params.output, x, words_full)  #将源嵌入映射到目标空间后按原格式写回到新的文本中
    save_matrix(params.output[:-4] + "_matrix.txt",  R)  #将映射矩阵保存下来