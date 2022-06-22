# encoding=utf-8
import torch
from knrm_pytorch.src.utils.utils import loadyaml
from knrm_pytorch.src.dataprocessors.tokenizer import Segment_jieba
from knrm_pytorch.src.dataprocessors.w2v import Embedding
from knrm_pytorch.src.dataprocessors.vocab import Vocab
# from knrm_pytorch.src.models.cknrm import CKNRM
from knrm_pytorch.src.utils.logger import setlogger
from knrm_pytorch.src.dataprocessors.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from knrm_pytorch.src.train import train
from knrm_pytorch.src.test import test
import os
from knrm_pytorch.src.models.knrm import KNRM
import pandas as pd
import numpy as np
#
def weight():
    data=pd.read_excel(r'D:\my paper aa\train\train_trans_labeled_token2.xlsx')
    label=[]
    for i in range(len(data)):
        label.append(data['biaoqian'][i])
    class_sample_count = np.array([len(np.where(label == t)[0]) for t in np.unique(label)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in label])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

def train_model():
    config = loadyaml(r'f:\Anaconda\envs\testenv\Lib\site-packages\knrm_pytorch\conf\cknrm.yaml')  ##加载配置文件 包含各个路径信息及模型配置信息
    logger = setlogger(config)
    torch.manual_seed(config['seed'])  ##固定随机种子
    config['device'] = 'cpu'
    print(f"device:{config['device']}")

    print(f"Begin to load the embeding")
    embedding = Embedding(r'G:\my paper\pre_enb_vector\all_align_quchong.vec',logger=logger)  ##加载嵌入向量,字典类型

    print(f"Begin to build segment")
    segment = Segment_jieba()  ##返回字典类型{'token':['我们','的','歌','呀']}


    print(f"Begin to build vec_dic")  #调用了embedding.true_word()方法，segment.seg()方法
    vocab = Vocab(config['datapath'], segment, embedding)  ##path可以忽略，没用
    print(f"vec_dic length: {len(vocab)}")  ##构造word2idx字典{'我们':0，'的':1，'歌':2}
    # a=vocab.idx2word
    # a=embedding.w2v.idx2word
    print(f"Begin to build dataset")  #调用了segment.seg()方法，vocab.word2idx字典
    train_dataset = Dataset(r'D:\my paper aa\train\train_trans_labeled_token2.xlsx', segment, vocab.word2idx, config)  ##从word2idx中获取每句话的id号码，不够最大长度的补0
    test_dataset = Dataset(r'D:\my paper aa\test_new\test_set_split1.xlsx', segment, vocab.word2idx, config)  ##评估集的id号码获取及补充

    print(f"Begin to buidl train_loader")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, sampler=weight())  ##转换成pytorch需要的数据输入
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)  ##每次读出样本batch_size=48个

    print(f"Init the model")  ##调用了embeding.get_vectors方法以及len(self.embedding.w2v)得到向量的长度
    model = KNRM(config, embedding).to(config['device'])

    print(f"Begin to train ......")
    train(train_loader, test_loader, model, config, logger)

###注意测试集的进行应该在训练集之后，因为要选出最好的一个训练模型来
def test_model():
    config = loadyaml(r'f:\Anaconda\envs\testenv\Lib\site-packages\knrm_pytorch\conf\cknrm.yaml')
    logger = setlogger(config)
    torch.manual_seed(config['seed'])
    config['device'] = 'cpu'

    print(f"Begin to load the embeding")
    embedding = Embedding(r'G:\my paper\pre_enb_vector\all_align_quchong.vec',logger=logger)

    print(f"Begin to build segment")
    segment = Segment_jieba()

    print(f"Begin to build vocab")
    vocab = Vocab(config['datapath'], segment, embedding)

    print(f"Begin to build dataset")
    test_dataset = Dataset(r'‪D:\my paper aa\test_new\test_set_split2.xlsx', segment, vocab.word2idx, config)  ##测试集，源代码又重新使用的验证集
    # print(train_dataset[3])

    print(f"Begin to buidl train_loader")
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    print(f"Init the model")  ###注意测试集的进行应该在训练集之后，因为要选出最好的一个训练模型来
    model = KNRM(config, embedding)
    if os.path.exists(r'D:\my paper aa\knrm_torch_result\cknrm_model.pt'):
        checkpoint = torch.load(r'D:\my paper aa\knrm_torch_result\cknrm_model.pt')
        model.load_state_dict(checkpoint['model'])
    test(test_loader, model, logger)


if __name__ == '__main__':
    # train_model()
    # 0.9788262990696657
    test_model()