import torch.nn as nn
import numpy as np
import torch
import time

def get_hierarchy_relations(label_map):
    hierar_relations = {}
    for key, value in label_map.items():
        label_layer = key.split('.')
        if len(label_layer) == 1:
            hierar_relations[value] = []
        elif len(label_layer) == 2:
            my_father = label_map[label_layer[0]]
            hierar_relations[my_father].append(value)
            hierar_relations[value] = []
        elif len(label_layer) == 3:
            my_father = label_map[label_layer[0] + '.' + label_layer[1]]
            hierar_relations[my_father].append(value)
            hierar_relations[value] = []
        elif len(label_layer) == 4:
            my_father = label_map[label_layer[0] + '.' + label_layer[1] + '.' + label_layer[2]]
            hierar_relations[my_father].append(value)

    return hierar_relations   # 返回树的连接关系的有关字典如[1:{11,5,9,4}....]


class ClassificationLoss(torch.nn.Module):
    def __init__(self, config, label_map, class_weight=None):     # 惩罚力度

        super(ClassificationLoss, self).__init__()
        self.config = config
        # print('class_weight, len : ' + str(len(class_weight)))
        # print(class_weight)

        if class_weight != None:
            self.loss_fn = torch.nn.BCEWithLogitsLoss(weight=class_weight.to(config.device))
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight.to(config.device))
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        # 树形结构的字典表示父根节点和子节点  {11：[1,8,36,23]}
        self.recursive_relation = get_hierarchy_relations(label_map)
        self.recursive_penalty = config.hierar_penalty
        self.use_recursive_penalty = config.use_hierar_penalty  # 默认1e-6
        self.device = config.device


        # 构建一个set
        self.relation_set = set()
        self.father_num = 0
        for key, value in self.recursive_relation.items():
            self.relation_set.add(key)
            self.father_num += 1
        # self.pdist = torch.nn.PairwiseDistance(p=2)

    def _recursive_regularization(self, params):
        """
        recursive regularization: constraint on the parameters of classifier among parent and children
        :param params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
        :param device: torch.device -> config.train.device_setting.device
        :return: loss -> torch.FloatTensor, ()
        """
        recursive_loss = 0.0
        for i in range(len(params)):  # len(params) = 40 类别数量
            if i + 1 >= self.father_num:
                break
            child_ids = self.recursive_relation[i]
            # if not child_ids:
            #     continue
            child_ids_list = torch.tensor(child_ids, dtype=torch.long, device=self.device)
            # 在0维上（行）索引出对应child-list的数据，也就是取出子节点相关参数的部分
            child_params = torch.index_select(params, 0, child_ids_list)
            # 在0维上（行）索引出对应的数据，也就是取出这些子节点的父节点相关参数的部分
            parent_params = torch.index_select(params, 0, torch.tensor(i, device=self.device))
            # parent_params = parent_params.repeat(child_ids_list.size()[0], 1)  # repeat(第一维度复制的次数, 第二维度复制的次数) 把第一维复制4次
            diff_paras = torch.sub(parent_params, child_params)
            # o_distance = self.pdist(parent_params, child_params)
            # recursive_loss += torch.sum(o_distance)
            recursive_loss += 0.5 * torch.sum(torch.mul(diff_paras, diff_paras))
            # recursive_loss += 0.00001

        return recursive_loss  # 一个数

    def forward(self, logits, targets, recursive_params):

        if self.use_recursive_penalty:
            loss = self.loss_fn(logits, targets) + \
                   self.recursive_penalty * self._recursive_regularization(recursive_params)
        else:
            loss = self.loss_fn(logits, targets)
        return loss