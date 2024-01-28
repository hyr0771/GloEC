import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

'''Attention Is All You Need'''
class Config_old(object):

    """配置参数"""
    def __init__(self):
        # 训练信息
        self.model_name = 'Transformer'
        self.data_name = 'model_data1'
        self.data_path = '../Data/model_data1.csv'   #文件路径
        self.k_fold = 10                # 先默认是10折交叉验证
        self.num_epoch = 100            # epoch数
        self.batch_size = 64            # mini-batch大小
        self.pad_size = 1000            # 每句话处理成的长度(短填长切)，这个根据自己的数据集而定
        self.learning_rate = 0.001      # 学习率
        self.PAD = '<PAD>'              #未知字与padding符号

        # 模型参数
        self.dropout = 0.1              # dropout的比例
        self.num_classes = 346          # 类别数,用于线性层输出   346表示所有的父子节点种类，即len(label_map)
        self.embed = 15                 # 字向量维度
        self.dim_model = 15             # 需要与embed一样
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 3               # 多头注意力，注意需要能整除dim_model
        self.num_encoder = 2            # 使用两个Encoder，尝试6个encoder发现存在过拟合，毕竟数据集量比较少（10000左右），可能性能还是比不过LSTM


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(config.num_encoder)])

        self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)
        # self.fc2 = nn.Linear(config.last_hidden, config.num_classes)
        # self.fc1 = nn.Linear(config.dim_model, config.num_classes)

    def forward(self, x):
        out = self.embedding(x[0])
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out


def get_attn_pad_mask(seq_q, seq_k):    # B站视频源码解读
    batch_size, len_q = seq_q.size()    ### [1,5]    seq_q: [batch_size,sen_len]
    batch_size, len_k = seq_k.size()    # eq(zero) is PAD token
    # 列mask
    l_mask = seq_k.data.eq(20).unsqueeze(1).expand(batch_size, len_q, len_k)  # batch_size x 1 x len_k(=len_q), one is masking
    # print(l_mask.size())
    n_l_mask = l_mask.cpu().numpy()
    # print(n_l_mask.shape)     # tensor输出维度用tensor.size(),numpy输出维度用np.shape
    n_attn_mask = np.logical_or(n_l_mask, n_l_mask.transpose(0, 2, 1))      # 或运算上一个转置
    pad_attn_mask = torch.tensor(n_attn_mask)
    # mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    return pad_attn_mask  # batch_size x len_q x len_k    ###expand()函数，将tensor变形为括号内的维度。expand到的维度 将第一行的数据重复就行了


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout, device):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, device, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x, attn_mask):
        out, attn = self.attention(x, attn_mask)
        out = self.feed_forward(out)
        return out, attn  # 什么维度的？


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        # **表示幂运算，//表示整除并向下取整
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  # 2 表示步进
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)   #每个batc都会被添加
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k, scale=None):
        '''
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        scores = torch.matmul(Q, K.transpose(3, 2)) /math.sqrt(d_k)    #K进行转制
        if scale:
            attention = scores * scale                   #？？？？？？
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        # scores = scores.masked_fill_(mask == 0, -1e9)   应该用哪一种？？

        # scores = scores.masked_fill_(mask=attn_mask, value=torch.tensor(-1e9))  # 暂时隐藏掉

        p_attn = F.softmax(scores, dim=-1)
        # if dropout is not None:
        #     p_attn = dropout(p_attn)
        context = torch.matmul(scores, V)    #matmul 高纬度矩阵乘法
        return torch.matmul(p_attn, V), p_attn     # 返回Z


class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model, num_head, device, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert d_model % num_head == 0
        self.d_k = d_model // self.num_head
        self.W_Q = nn.Linear(d_model, num_head * self.d_k)  # 线性层，dim_model表示输入神经元个数，参数2表示输出神经元个数
        self.W_K = nn.Linear(d_model, num_head * self.d_k)
        self.W_V = nn.Linear(d_model, num_head * self.d_k)
        self.attention = Scaled_Dot_Product_Attention()    # 返回Z矩阵
        self.linear = nn.Linear(num_head * self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, attn_mask):
        batch_size = x.size(0)   # 非固定可能不完整，因为分数据的时候剩下的可能不够分
        x = x.to(torch.float32)
        Q = self.W_Q(x)
        # print(self.W_Q.parameters())
        K = self.W_K(x)
        V = self.W_V(x)
        # 分头怎么理解？
        Q = Q.view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)   #view函数用来重构张量大小，-1表示自动补齐张量
        K = K.view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)   #torch.Size([8, 1000, 25])、torch.Size([8, 5, 1000, 5])
        V = V.view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)   #transpose维度变换，把第一维与第二维互换，原本维度（batch、L、num_head、d_k），变换之后为（batch、num_head、L、d_k）


        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子

        # padding_mask分头
        if not attn_mask is None:
            test_mask = attn_mask.unsqueeze(1)
            mask = attn_mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)  # unsequeeze（1）在第二维增加一个维度，下标从0开始。
            mask = mask.to(self.device)
        else:
            mask = None

        context, attn = self.attention(Q, K, V, mask, self.d_k, scale)   # context表示Z矩阵

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.d_k)
        # 原来
        # out = self.fc(context)
        # out = self.dropout(out)
        # out = out + x  # 残差连接
        # out = self.layer_norm(out)
        # return out
        out = self.linear(context)
        return self.layer_norm(out + x), attn   # 残差后进行layrtnormal,接下来进行前馈网络


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Transformer_model(nn.Module):
    def __init__(self, config, label_map):
        super(Transformer_model, self).__init__()
        if not config.use_biovec:
            self.embedding = nn.Embedding(21, config.embedding_dimension, padding_idx=20)  # 共21个字符包括0，其中第21个为pad，编号为20

        self.postion_embedding = Positional_Encoding(config.embedding_dimension, config.feature_max_char_len, config.OldTransformer.encoder_dropout, config.device)
        self.encoder = Encoder(config.OldTransformer.dim_model, config.OldTransformer.n_head, config.OldTransformer.hidden, config.OldTransformer.encoder_dropout, config.device)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.OldTransformer.n_layers)])   # 多次Encoder

        # 计算纯属亚子类的类别有多少个
        len_num = 0
        for key, value in label_map.items():
            num = str(key).replace('\n', '').replace(' ', '')
            num_list = num.split('.')
            if len(num_list) == 3:
                len_num += 1

        self.linear = nn.Linear(config.feature_max_char_len * config.OldTransformer.dim_model, len_num)

    def forward(self, x, length):
        # x = torch.as_tensor(x).to(torch.int64)
        # 先获得对应的padding_mask
        # enc_self_attn_mask = get_attn_pad_mask(x, x)
        enc_self_attn_mask = None

        # out = self.embedding(x)
        out = self.postion_embedding(x)
        enc_self_attns = []
        for encoder in self.encoders:
            out, enc_self_attn = encoder(out, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        out = out.view(out.size(0), -1)  # 将三维张量reshape成二维，然后直接通过全连接层将高维数据映射为classes
        # out = torch.mean(out, 1)    # 也可用池化来做，但是效果并不是很好
        out = self.linear(out)
        # softmax一下??好像不需要，交叉熵里做了？
        #soft_out = F.softmax(out, dim=-1)
        return out
