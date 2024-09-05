import json
import os
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
from HiAGM import selfAttention, get_probability

class ESM_Model(nn.Module):
    def __init__(self, config, label_map,  drop_out=0.25):
        super(ESM_Model, self).__init__()
        self.config = config
        self.label_map = label_map
        self.class_num = config.label_length[config.layer - 1]


        self.fc1 = nn.Linear(1280, 512)
        self.ln1 = nn.LayerNorm(512)
        self.attention = selfAttention(4, 512, 128)  # 4头，输入为512维度，输出为128维 b*1*128
        self.structureEncoder = StructureEncoder(config, label_map)

        self.classifier = nn.Linear(len(self.label_map)*16, len(self.label_map))
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.dropout(x)
        x = torch.relu(x)
        x = x.view(x.shape[0], 1, -1)
        x = self.attention(x)
        x = self.dropout(x)

        # 输入结构编码器
        label_wise_text_feature = self.structureEncoder(x)

        # 连接全连接层分类
        x = self.classifier(label_wise_text_feature)
        return x



class StructureEncoder(nn.Module):
    def __init__(self, config, label_map):
        super(StructureEncoder, self).__init__()

        self.config = config
        self.label_map = label_map

        # 维度变换
        self.transformation = nn.Linear(128, len(self.label_map) * 16)
        # dropout 对于上面两个线性层的dropout
        self.transformation_dropout = nn.Dropout()

        # 先验概率
        # 获取先验概率文件
        self.hierarchy_dic = get_probability(label_map)  # 有概率的边
        self.node_prob_from_parent = np.zeros((len(self.label_map), len(self.label_map)))
        self.node_prob_from_child = np.zeros((len(self.label_map), len(self.label_map)))

        # 制作先验概率矩阵
        for p in self.hierarchy_dic.keys():
            if p == 'Root':
                continue
            for c in self.hierarchy_dic[p]:
                # self.hierarchy_id_prob[self.label_map[p]][self.label_map[c]] = self.hierarchy_prob[p][c]
                self.node_prob_from_child[int(self.label_map[p])][int(self.label_map[c])] = 1.0   # 子节点到父节点的概率必定为1
                self.node_prob_from_parent[int(self.label_map[c])][int(self.label_map[p])] = 1.0  # 父节点到子节点的概率由概率表决定,但是我们的方法不需要概率
        #  node_prob_from_parent: row means parent, col refers to children  # 行为爸，列为子

        # 传入HierarchyGCN时， 数据的特征是(batch * 类别数 * 特征维度)  特征维度256维
        if config.use_GCN:
            self.gcn_layer = nn.ModuleList([
                MyGCNModule(config=config,
                            from_child_matrix=self.node_prob_from_child,
                            from_parent_matrix=self.node_prob_from_parent,
                            in_channels=16, out_channels=16)
                for _ in range(config.gcn_layer)])

    def forward(self, inputs):
        # 维度变换 b*1*128 --> b*class*32
        x = self.transformation_dropout(self.transformation(inputs)) # transformation是线性层，transformation_dropout是一个dropout层
        x = x.view(x.shape[0], len(self.label_map), -1)


        # 加入结构特征
        # b*class*32 --> b*class*32
        if self.config.use_GCN:
            for i, gc in enumerate(self.gcn_layer):
                x = self.gcn_layer[i](x)


        # 变换维度然后返回
        # logits = self.classifier_linear(label_wise_text_feature.view(label_wise_text_feature.shape[0], -1))
        label_wise_text_feature = x.view(x.shape[0], -1)
        return label_wise_text_feature

class MyGCNModule(nn.Module):
    def __init__(self,
                 config,
                 from_child_matrix,  # 子到父概率阵
                 from_parent_matrix,  # 父到子概率阵
                 in_channels,   # 输入特征的维度 (b * label_len * in_dim)  这里是32
                 out_channels):


        super(MyGCNModule, self).__init__()
        self.config = config
        self.in_channels = in_channels  # 输入图节点的特征数
        self.out_channels = out_channels  # 输出图节点的特征数

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
        # 是否要乘上概率矩阵
        # self.up_adj_t = torch.mm(self.up_degree_2, self.up_adj).mm(torch.Tensor(from_parent_matrix)).mm(self.up_degree_2).to(self.config.device)
        self.up_adj_t = torch.mm(self.up_degree_2, self.up_adj).mm(self.up_degree_2).to(self.config.device)

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
        # self.attention = selfAttention(4, 32, 32)  # 4头，输入为512维度，输出为128维 b*1*128
        self.init_parameters()  # 重置上面的参数？


    def init_parameters(self):
        nn.init.xavier_uniform_(self.down_weight)
        nn.init.xavier_uniform_(self.down_bias)
        nn.init.xavier_uniform_(self.up_weight)
        nn.init.xavier_uniform_(self.up_bias)
        nn.init.xavier_uniform_(self.loop_weight)
        nn.init.xavier_uniform_(self.loop_bias)

    def forward(self, inputs):  # (291, 32)
        # a_input = self.attention(inputs)
        message_ = torch.zeros_like(inputs).to(self.config.device)  # batch, N, in_dim

        # 自下而上
        # 1.计算HW
        down_x = torch.matmul(inputs, self.down_weight)  # num_nodes, out_channels
        # 2.计算D^(-0.5)AD^(-0.5)HW
        down_output = torch.matmul(self.down_adj_t, down_x)  # 计算
        # 3.添加偏置
        down_output = down_output + self.down_bias.flatten()
        # message_ = torch.zeros_like(down_output).to(self.config.device)  # batch, N, in_dim
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

        # 残差连接
        x = inputs + self.activation(message_)  # 不一定有用波

        return x



if __name__ == '__main__':
    print('66')
