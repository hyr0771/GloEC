import json
import os
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


def get_probability(label_map):
    hiera_dic = {}
    hiera_dic['Root'] = []

    for key, value in label_map.items():
        layer = str(key).split('.')
        if len(layer) == 1:
            hiera_dic['Root'].append(key)
            hiera_dic[key] = []
        elif len(layer) == 2:
            hiera_dic[layer[0]].append(key)
            hiera_dic[key] = []
        elif len(layer) == 3:
            layer2_key = layer[0] + '.' + layer[1]
            hiera_dic[layer2_key].append(key)
            hiera_dic[key] = []
        elif len(layer) == 4:
            layer3_key = layer[0] + '.' + layer[1] + '.' + layer[2]
            hiera_dic[layer3_key].append(key)

    # 不需要计算概率了
    # for key, _ in pro_hiera_dic.items():
    #     total_num = 0
    #     for _, layer_total_num in pro_hiera_dic[key].items():
    #         total_num += layer_total_num
    #
    #     for layer_key, layer_num in pro_hiera_dic[key].items():
    #         pro_hiera_dic[key][layer_key] = layer_num/total_num


    return hiera_dic


class HiAGM(nn.Module):
    def __init__(self, config, label_dic):
        """
        Hierarchy-Aware Global Model class
        :param config: helper.configure, Configure Object
        """
        super(HiAGM, self).__init__()
        self.config = config

        self.attention = selfAttention(4, 64, 64)  # 4头，输入为100维度，输出为64维
        self.attention_dropout = nn.Dropout(p=0.2)

        # input_size:输入特征的维度， hidden_size:输出特征的维度
        self.num_layers = 1
        self.hidden_size = 32
        self.bidirectional = True
        self.lstm = nn.LSTM(input_size=100, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, dropout=0.2, bidirectional=self.bidirectional)

        self.kernel_sizes = [2, 3, 4]   # 原来是[2, 3, 4]
        self.convs = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(torch.nn.Conv1d(in_channels=64, out_channels=64,
                kernel_size=kernel_size, padding=kernel_size // 2))

        self.pad = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.pad.append(CustomPad(kernel_size))
        # self.conv1 = nn.Conv1d(128, 64, kernel_size=3)  # 1600个参数

        # 模型类型的选择
        self.hiagm = HiAGMTP(config=config, label_map=label_dic)

    def forward(self, batch):  # 3*100
        # x = self.attention(batch)    # 3*64
        # x = self.attention_dropout(x)
        # x = x.transpose(1, 2)   # 64*3

        # 初始化状态
        if self.bidirectional:
            h0 = torch.randn(self.num_layers * 2, batch.shape[0], self.hidden_size).to(self.config.device)
            c0 = torch.randn(self.num_layers * 2, batch.shape[0], self.hidden_size).to(self.config.device)
        else:
            h0 = torch.randn(self.num_layers, batch.shape[0], self.hidden_size).to(self.config.device)
            c0 = torch.randn(self.num_layers, batch.shape[0], self.hidden_size).to(self.config.device)

        x, _ = self.lstm(batch, (h0, c0))  # 3*64
        x = self.attention(x)
        x = self.attention_dropout(x)

        x = x.transpose(1, 2)
        conv_text_outputs = []
        # 进行卷积
        for index, conv in enumerate(self.convs):
            # 进行填充
            # pad_x = self.pad[index](x)
            # F.pad(x, pad, mode='constant', value=0)
            convolution = torch.relu(conv(x))
            # 取最大值还是取均值咯?????取均值好一点
            mean_x = torch.mean(convolution, dim=2)
            # top_x = torch.topk(convolution, k=1, dim=1)[0].view(x.size(0), -1)
            # top_x = top_x.unsqueeze(1)
            mean_x = mean_x.unsqueeze(1)
            conv_text_outputs.append(mean_x)

        logits = self.hiagm(conv_text_outputs)

        return logits

class StructureEncoder(nn.Module):
    def __init__(self, config, label_map):
        super(StructureEncoder, self).__init__()

        self.label_map = label_map
        # 先验概率
        # 获取先验概率文件
        self.hierarchy_prob_dic = get_probability(config)
        self.node_prob_from_parent = np.zeros((len(self.label_map), len(self.label_map)))
        self.node_prob_from_child = np.zeros((len(self.label_map), len(self.label_map)))

        # 制作先验概率矩阵
        for p in self.hierarchy_prob_dic.keys():
            if p == 'Root':
                continue
            for c in self.hierarchy_prob_dic[p].keys():
                # self.hierarchy_id_prob[self.label_map[p]][self.label_map[c]] = self.hierarchy_prob[p][c]
                self.node_prob_from_child[int(self.label_map[p])][int(self.label_map[c])] = 1.0   # 子节点到父节点的概率必定为1
                self.node_prob_from_parent[int(self.label_map[c])][int(self.label_map[p])] = self.hierarchy_prob_dic[p][c] # 父节点到子节点的概率由概率表决定
        #  node_prob_from_parent: row means parent, col refers to children  # 行为爸，列为子

        # 这里这个model还可以是TreeLSTM 参考源代码就行
        self.model = MyGCNModule(config=config, num_nodes=len(self.label_map),
                                from_child_matrix=self.node_prob_from_child,
                                from_parent_matrix=self.node_prob_from_parent,
                                in_dim=32,          # 传入HierarchyGCN时， 数据的特征是(batch * 类别数 * 特征维度)  特征维度256维
                                dropout=0.1)

    def forward(self, inputs):
        return self.model(inputs)


class HiAGMTP(nn.Module):
    def __init__(self, config, label_map):
        """
        Hierarchy-Aware Global Model : (Serial) Text Propagation Variant
         :param config: helper.configure, Configure Object
        :param label_map: helper.vocab.Vocab.v2i['label'] -> Dict{str:int}
        """
        super(HiAGMTP, self).__init__()

        self.config = config
        self.label_map = label_map

        self.structure_Encoder = StructureEncoder(config=config, label_map=label_map)

        # linear transform
        # self.transformation = nn.Linear(64*3, len(self.label_map) * 32)
        # 改
        self.transformation = nn.Linear(64*3, 7 * 32)

        # # classifier
        # self.classifier_linear = nn.Linear(len(self.label_map) * 32, len(self.label_map))
        # 改
        self.classifier_linear = nn.Linear(7 * 32, 7)

        # dropout
        self.transformation_dropout = nn.Dropout(p=0.15)
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, text_feature):
        """
        forward pass of text feature propagation
        :param text_feature ->  torch.FloatTensor, (batch_size, K0, text_dim)
        :return: logits ->  torch.FloatTensor, (batch, N)
        """
        # 经过一些列处理
        text_feature = torch.cat(text_feature, 1)  # b*3*64
        text_feature = text_feature.view(text_feature.shape[0], -1)  # b*(3*64)

        # b*(32*213)
        text_feature = self.transformation_dropout(self.transformation(text_feature)) # transformation是线性层，transformation_dropout是一个dropout层
        # text_feature = text_feature.view(text_feature.shape[0],
        #                                  len(self.label_map), -1)   # b*213*32
        # 改
        text_feature = text_feature.view(text_feature.shape[0], 7, -1)   # b*213*32

        # label_wise_text_feature = self.structure_Encoder(text_feature)   #这里指hieira-GCN

        # logits = self.dropout(self.classifier_linear(label_wise_text_feature.view(label_wise_text_feature.shape[0], -1)))
        # 改
        logits = self.dropout(self.classifier_linear(text_feature.view(text_feature.shape[0], -1)))
        # logits = self.dropout(self.linear(text_feature.view(text_feature.shape[0], -1)))
        return logits

# 封装好料 别人的层级结构
class HierarchyGCNModule(nn.Module):
    def __init__(self,
                 config,
                 num_nodes,   # 类别数量40
                 from_child_matrix,  # 子到父概率阵
                 from_parent_matrix,  # 父到子概率阵
                 in_dim,   # 输入特征的维度 (b * label_len * in_dim)  这里是32
                 dropout):

        super(HierarchyGCNModule, self).__init__()
        self.config = config
        #  bottom-up child sum
        in_prob = from_child_matrix  # 子到父概率
        # 用了Parameter来定义表示这是一个需要训练的参数，加入到parameter()这个迭代器中去
        # torch.Tensor是一种包含单一数据类型元素的多维矩阵
        # adj_matrix的内容基本就是in_prob的内容
        self.adj_matrix = Parameter(torch.Tensor(in_prob))  # 子到父概率,领接矩阵
        self.edge_bias = Parameter(torch.Tensor(num_nodes, in_dim))  # 边缘偏差（40,300）
        self.gate_weight = Parameter(torch.Tensor(in_dim, 1))  # 门权重（300,1）
        self.bias_gate = Parameter(torch.Tensor(num_nodes, 1))  # 偏差门（40,1）
        self.activation = nn.ReLU()  # 引入非线性
        # origin_adj基本上可以认为就是 in_adj
        self.origin_adj = torch.Tensor(np.where(from_child_matrix <= 0, from_child_matrix, 1.0)).to(self.config.device) # 子到父概率矩阵，原生领接矩阵不参与后续梯度计算
        # top-down: parent to child  # 父到子
        self.out_adj_matrix = Parameter(torch.Tensor(from_parent_matrix))  # 父到子概率矩阵
        self.out_edge_bias = Parameter(torch.Tensor(num_nodes, in_dim))  # out边缘偏差（40,300）
        self.out_gate_weight = Parameter(torch.Tensor(in_dim, 1))  # out门权重（300,1）
        self.out_bias_gate = Parameter(torch.Tensor(num_nodes, 1))  # out偏差门（40,1）

        self.loop_gate = Parameter(torch.Tensor(in_dim, 1))  # 循环门（300,1）
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()  # 重置上面的参数？

    # 初始化一些矩阵
    def reset_parameters(self):
        """
        initialize parameters
        """
        for param in [self.gate_weight, self.loop_gate, self.out_gate_weight]:
            nn.init.xavier_uniform_(param)  # 将那3个变量进行xavier初始化，符合一定分布
        for param in [self.edge_bias, self.out_edge_bias, self.bias_gate]:
            nn.init.zeros_(param)  # 使用0对tensor赋值

    # 这里都干了些啥
    def forward(self, inputs):
        """
        :param inputs: torch.FloatTensor, (batch_size, N, in_dim)
        :return: message_ -> torch.FloatTensor (batch_size, N, in_dim)
        """
        # 信息聚合

        h_ = inputs  # batch, N, in_dim
        # 产生一个和h_维度相同的全0矩阵
        message_ = torch.zeros_like(h_).to(self.config.device)  # batch, N, in_dim

        # 信息1
        # ukj = akj*vj + b
        # h_可以看成是v？节点向量
        # torch.matmul表示矩阵叉乘  tensor * tensor 表示矩阵点乘(对应元素相乘) 至少第一维度要一样，然后顺序乘过去
        h_in_ = torch.matmul(self.origin_adj * self.adj_matrix, h_)  # batch, N, in_dim
        in_ = h_in_ + self.edge_bias  # (class_num , in_dim) + (class_num , in_dim)
        # in_ = in_
        # N,1,dim
        in_gate_ = torch.matmul(h_, self.gate_weight)  # (class_num , in_dim) * (in_dim , 1)
        # N, 1
        in_gate_ = in_gate_ + self.bias_gate  # (class_num , 1) + (class_num , 1)
        in_ = in_ * torch.sigmoid(in_gate_)  # (class_num , in_dim) 点乘 (class_num , 1) = (class_num , in_dim)
        in_ = self.dropout(in_)
        message_ += in_  # batch, N, in_dim

        # 信息2
        # self.origin_adj.transpose(0, 1) 是 self.origin_adj矩阵的转置，表示父到子的路径存在情况
        h_output_ = torch.matmul(self.origin_adj.transpose(0, 1) * self.out_adj_matrix, h_)
        out_ = h_output_ + self.out_edge_bias
        out_gate_ = torch.matmul(h_, self.out_gate_weight)
        out_gate_ = out_gate_ + self.out_bias_gate
        out_ = out_ * torch.sigmoid(out_gate_)
        out_ = self.dropout(out_)
        message_ += out_

        # 信息3
        loop_gate = torch.matmul(h_, self.loop_gate)
        loop_ = h_ * torch.sigmoid(loop_gate)
        loop_ = self.dropout(loop_)
        message_ += loop_

        return self.activation(message_)  # ReLu()

# 自己改一改层级结构
class MyGCNModule(nn.Module):
    def __init__(self,
                 config,
                 num_nodes,   # 类别数量40
                 from_child_matrix,  # 子到父概率阵
                 from_parent_matrix,  # 父到子概率阵
                 in_dim,   # 输入特征的维度 (b * label_len * in_dim)  这里是32
                 dropout):


        super(MyGCNModule, self).__init__()
        self.config = config
        self.in_channels = 32  # 输入图节点的特征数
        self.out_channels = 32  # 输出图节点的特征数

        # 下到上
        # 1.获取邻接矩阵A
        self.down_adj = torch.Tensor(np.where(from_child_matrix <= 0, from_child_matrix, 1.0))
        # 2.添加自环
        self.down_adj = self.down_adj + torch.eye(self.down_adj.shape[0])
        # 3.获取度矩阵D
        self.down_degree = self.down_adj.sum(axis=1)
        # 4.获取D^(-0.5)
        self.down_degree_2 = torch.torch.diag_embed(torch.pow(self.down_degree, -0.5).flatten())
        # 5.计算D^(-0.5)AD^(-0.5)
        self.down_adj_t = torch.mm(self.down_degree_2, self.down_adj).mm(self.down_degree_2).to(self.config.device)

        # 上到下
        self.up_adj = torch.Tensor(np.where(from_parent_matrix <= 0, from_parent_matrix, 1.0))
        self.up_adj = self.up_adj + torch.eye(self.up_adj.shape[0])
        self.up_degree = self.up_adj.sum(axis=1)
        self.up_degree_2 = torch.torch.diag_embed(torch.pow(self.up_degree, -0.5).flatten())
        self.up_adj_t = torch.mm(self.up_degree_2, self.up_adj).mm(torch.Tensor(from_parent_matrix)).mm(self.up_degree_2).to(self.config.device)

        # 自循环
        self.loop_adj = torch.Tensor(np.where(from_parent_matrix <= 0, from_parent_matrix, 1.0))
        self.loop_adj = torch.eye(self.loop_adj.shape[0])
        self.loop_degree = self.loop_adj.sum(axis=1)
        self.loop_degree_2 = torch.torch.diag_embed(torch.pow(self.loop_degree, -0.5).flatten())
        self.loop_adj_t = torch.mm(self.loop_degree_2, self.loop_adj).mm(self.loop_degree_2).to(self.config.device)

        # 定义训练参数 θ
        self.down_weight = nn.Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
        self.down_bias = nn.Parameter(torch.FloatTensor(self.out_channels, 1))

        self.up_weight = nn.Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
        self.up_bias = nn.Parameter(torch.FloatTensor(self.out_channels, 1))

        self.loop_weight = nn.Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
        self.loop_bias = nn.Parameter(torch.FloatTensor(self.out_channels, 1))

        self.activation = nn.ReLU()
        self.init_parameters()  # 重置上面的参数？


    def init_parameters(self):
        nn.init.xavier_uniform_(self.down_weight)
        nn.init.xavier_uniform_(self.down_bias)
        nn.init.xavier_uniform_(self.up_weight)
        nn.init.xavier_uniform_(self.up_bias)
        nn.init.xavier_uniform_(self.loop_weight)
        nn.init.xavier_uniform_(self.loop_bias)


    # 这里都干了些啥
    def forward(self, inputs):  # (291, 32)
        message_ = torch.zeros_like(inputs).to(self.config.device)  # batch, N, in_dim

        # 自下而上
        # 1.计算HW
        down_x = torch.matmul(inputs, self.down_weight)  # num_nodes, out_channels
        # 2.计算D^(-0.5)AD^(-0.5)HW
        down_output = torch.matmul(self.down_adj_t, down_x)  # 计算
        # 3.添加偏置
        down_output = down_output + self.down_bias.flatten()
        message_ += down_output

        # 自上而下
        up_x = torch.matmul(inputs, self.up_weight)  # num_nodes, out_channels
        up_output = torch.matmul(self.up_adj_t, up_x)  # 计算
        up_output = up_output + self.up_bias.flatten()
        message_ += up_output

        # 自循环
        loop_x = torch.matmul(inputs, self.loop_weight)  # num_nodes, out_channels
        loop_output = torch.matmul(self.loop_adj_t, loop_x)  # 计算
        loop_output = loop_output + self.loop_bias.flatten()
        message_ += loop_output

        return self.activation(message_)

# 自定义扩充函数
class CustomPad(nn.Module):
    def __init__(self, kernel_sizes):
        super(CustomPad, self).__init__()

        add_pad = (kernel_sizes - 1) // 2
        if kernel_sizes % 2 == 0:
            self.padding = (add_pad + 1, add_pad)
        else:
            self.padding = (add_pad, add_pad)

    def forward(self, x):
        return F.pad(x, self.padding, mode='constant', value=0)

class selfAttention(nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = torch.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)
        return context


if __name__ == '__main__':
    print('66')
