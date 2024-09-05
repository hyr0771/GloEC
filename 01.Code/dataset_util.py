import time
from datetime import timedelta

import numpy
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import json
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pickle

class My_Dataset(Dataset):
    def __init__(self, esm_tensor, label):

        self.esm_tensor = esm_tensor
        self.multi_label_onehot = label['one_hot']
        self.multi_label = label['three_label']
        self.layer_1_label = label['layer_1']
        self.layer_2_label = label['layer_2']
        self.layer_3_label = label['layer_3']
        self.layer_4_label = label['layer_4']

    def __len__(self):
        return len(self.esm_tensor)

    def __getitem__(self, idx):

        return (self.esm_tensor[idx], self.multi_label_onehot[idx], self.multi_label[idx],
                self.layer_1_label[idx], self.layer_2_label[idx], self.layer_3_label[idx],
                self.layer_4_label[idx])

class other_Dataset(Dataset):
    def __init__(self, esm_tensor, str_label, name_list):

        self.esm_tensor = esm_tensor
        self.str_label = str_label
        self.name_list = name_list

    def __len__(self):
        return len(self.esm_tensor)

    def __getitem__(self, idx):

        return (self.esm_tensor[idx], self.str_label[idx], self.name_list[idx])

def build_vocab(config):
    # 该函数返回的是所有序列中氨基酸出现的频率，按照高到低排列表示为0,1,2....
    print('构建新词表：' + config.data_name + '_CharVocab.json')
    enz_data = pd.read_csv(config.creat_vocab_data_path)
    sequence = enz_data.iloc[:, 2:3].values  # 获取序列

    ##构建出语料表
    vocab_dic = {}
    for seq in sequence:
        for word in seq[0]:
            vocab_dic[word] = vocab_dic.get(word, 0) + 1 

    # print(vocab_dic)
    vocab_list = sorted(vocab_dic.items(), key=lambda x: x[1], reverse=True)  # 排序后生成list
    # 获取list中前20即可
    temp_dic = {}
    for i in range(20):
        temp_dic[vocab_list[i][0]] = i + 1
    temp_dic[config.PAD] = 0
    vocab_dic = temp_dic
    # vocab_dic = {word_count[0]:idx for idx,word_count in enumerate(vocab_list)}
    print('整理与排序之后词表') 
    # print(vocab_dic)


    print(vocab_dic)
    return vocab_dic

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def get_label_map(config):
    if config.iskfold:
        with open(config.kfold_label_map, 'r+') as file:
            label_map = json.loads(file.read())  # 将json格式文件转化为python的字典文件
    else:
        with open(config.label_map_json, 'r+') as file:
            label_map = json.loads(file.read())  # 将json格式文件转化为python的字典文件

    layer_1_class, layer_2_class, layer_3_class, layer_4_class = 0, 0, 0, 0
    for key, value in label_map.items():
        hiera = len(str(key).split('.'))
        if hiera == 1:
            layer_1_class += 1
        if hiera == 2:
            layer_2_class += 1
        if hiera == 3:
            layer_3_class += 1
        if hiera == 4:
            layer_4_class += 1

    return label_map

def get_char_map(config):

    if (os.path.exists(config.data_path + config.data_name + '_CharVocab.json')):
        print('存在 ' + config.data_name + '_CharVocab.json 文件，正在读取')
        with open(config.data_path + config.data_name + '_CharVocab.json', 'r+') as file:
            char_map = file.read()
            char_map = json.loads(char_map)  # 将json格式文件转化为python的字典文件
    else:
        print('不存在 ' + config.data_name + '_CharVocab.json 文件，正在创建')
        char_map = build_vocab(config)
        char_json = json.dumps(char_map)  # 转化为json格式文件
        with open(config.data_path + config.data_name + '_CharVocab.json', 'w') as file:
            file.write(char_json)

    return char_map

def get_result(config):
    with open(config.model_save_path + config.model_type + '/result.json', 'r+') as file:
        result_dit = json.loads(file.read())  # 将json格式文件转化为python的字典文件
    return result_dit

def save_result(dict, config):
    temp_json = json.dumps(dict)  # 转化为json格式文件
    with open(config.model_save_path + config.model_type + '/result.json', 'w') as file:
        file.write(temp_json)

def get_type_dataloader(config, label_map, type='train' ,kfold_eval_index_path=None):
    if type == 'train':
        train_esm_data_path = config.data_path + 'esm_f_train.pt'
        esm_tensor, label = \
            process_esm_data(config.train_data_path, config, label_map, train_esm_data_path)  # 获取数据
        train_data = My_Dataset(esm_tensor, label)
        train_data_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, collate_fn=collate_fn,
                                       shuffle=True, num_workers=4, pin_memory=True)

        eval_esm_data_path = config.data_path + 'esm_f_eval.pt'
        esm_tensor, label = \
            process_esm_data(config.eval_data_path, config, label_map, eval_esm_data_path)  # 获取数据
        validate_data = My_Dataset(esm_tensor, label)
        validate_data_loader = DataLoader(dataset=validate_data, batch_size=config.batch_size, collate_fn=collate_fn,
                                          shuffle=False, num_workers=4, pin_memory=True)

        return train_data_loader, validate_data_loader

    elif type == 'test':
        test_esm_data_path = '../Data/uniport_2022_5/esm_f_time_test.pt'
        esm_tensor, str_label, name_list = \
            process_other_esm_data('../Data/uniport_2022_5/f_time_test.csv', test_esm_data_path)  # 获取数据
        test_data = other_Dataset(esm_tensor, str_label, name_list)
        test_data_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, collate_fn=other_collate_fn,
                                      shuffle=False, num_workers=4, pin_memory=True)
        return test_data_loader

    ###### 注意全部用other
    elif type == 'price':
        price_esm_data_path = '../Data/price/price.pt'
        esm_tensor, str_label, name_list = \
            process_other_esm_data('../Data/price/price_to_my_model.csv', price_esm_data_path)  # 获取数据
        price_data = other_Dataset(esm_tensor, str_label, name_list)
        price_data_loader = DataLoader(dataset=price_data, batch_size=config.batch_size, collate_fn=other_collate_fn,
                                      shuffle=False, num_workers=4, pin_memory=True)
        return price_data_loader

    elif type == 'np':
        np_esm_data_path = '../Data/np_dataset/np_dataset.pt'
        esm_tensor, str_label, name_list = \
            process_other_esm_data('../Data/np_dataset/np_dataset.csv', np_esm_data_path)  # 获取数据
        np_data = other_Dataset(esm_tensor, str_label, name_list)
        np_data_loader = DataLoader(dataset=np_data, batch_size=config.batch_size, collate_fn=other_collate_fn,
                                      shuffle=False, num_workers=4, pin_memory=True)
        return np_data_loader

    elif type == 'isoform':
        isoform_esm_data_path = '../Data/isoform_dataset/isoform_only_dataset.pt'
        esm_tensor, str_label, name_list = \
            process_other_esm_data('../Data/isoform_dataset/isoform_only_dataset.csv', isoform_esm_data_path)  # 获取数据
        isoform_data = other_Dataset(esm_tensor, str_label, name_list)
        isoform_data_loader = DataLoader(dataset=isoform_data, batch_size=config.batch_size, collate_fn=other_collate_fn,
                                      shuffle=False, num_workers=4, pin_memory=True)
        return isoform_data_loader

    elif type == 'co':
        co_esm_data_path = '../Data/cofactor_dataset/revise_only_cofactor.pt'
        esm_tensor, str_label, name_list = \
            process_other_esm_data('../Data/cofactor_dataset/revise_only_cofactor.csv', co_esm_data_path)  # 获取数据
        co_data = other_Dataset(esm_tensor, str_label, name_list)
        co_data_loader = DataLoader(dataset=co_data, batch_size=config.batch_size, collate_fn=other_collate_fn,
                                      shuffle=False, num_workers=4, pin_memory=True)
        return co_data_loader

    elif type == 'nc':
        nc_esm_data_path = '../Data/nc_dataset/nc_dataset.pt'
        esm_tensor, str_label, name_list = \
            process_other_esm_data('../Data/nc_dataset/nc_dataset.csv', nc_esm_data_path)  # 获取数据
        nc_data = other_Dataset(esm_tensor, str_label, name_list)
        nc_data_loader = DataLoader(dataset=nc_data, batch_size=config.batch_size, collate_fn=other_collate_fn,
                                      shuffle=False, num_workers=4, pin_memory=True)
        return nc_data_loader

    elif type == 'kfold':
        all_kfold_data_path = '../Data/uniport_2022_5/0.5similarity_dataset/enzyme_0.5.pt'
        esm_tensor, str_label, name_list = \
            process_kfold_esm_data('../Data/uniport_2022_5/0.5similarity_dataset/enzyme_0.5.csv', all_kfold_data_path, kfold_eval_index_path)  # 获取数据
        kfold_eval_data = other_Dataset(esm_tensor, str_label, name_list)
        kfold_eval_data_loader = DataLoader(dataset=kfold_eval_data, batch_size=config.batch_size, collate_fn=other_collate_fn,
                                      shuffle=False, num_workers=4, pin_memory=True)
        return kfold_eval_data_loader


def get_weight(config, layer=3):
    with open(config.label_map_json, 'r+') as file:
        label_map = json.loads(file.read())  # 将json格式文件转化为python的字典文件
    with open(config.weight, 'r+') as file:
        label_num = json.loads(file.read())  # 每种类别的个数

    layer_1_class, layer_2_class, layer_3_class, layer_4_class = 0, 0, 0, 0
    for key, value in label_map.items():
        hiera = len(str(key).split('.'))
        if hiera == 1:
            layer_1_class += 1
        if hiera == 2:
            layer_2_class += 1
        if hiera == 3:
            layer_3_class += 1
        if hiera == 4:
            layer_4_class += 1

    # 计算权重
    layer_1_num, layer_2_num, layer_3_num, layer_4_num = {}, {}, {}, {}
    for key, value in label_num.items():
        child_split_len = len(str(key).split('.'))
        if child_split_len == 1:
            layer_1_num[label_map[key]] = label_num[key]
        elif child_split_len == 2:
            layer_2_num[label_map[key] - layer_1_class] = label_num[key]
        elif child_split_len == 3:
            layer_3_num[label_map[key] - layer_1_class - layer_2_class] = label_num[key]
        elif child_split_len == 4:
            layer_4_num[label_map[key] - layer_1_class - layer_2_class - layer_3_class] = label_num[key]
# 数据有问题，怎么那么多1的
    if layer == 1:
        weight_list = comput_weight(layer_1_num)
        class_num = layer_1_class
    elif layer == 2:
        weight_list = comput_weight(layer_2_num)
        class_num = layer_2_class
    elif layer == 3:
        weight_list = comput_weight(layer_3_num)
        class_num = layer_3_class

    # 全部类别的weight_list
    class_weight_list = []
    class_weight_list += comput_weight(layer_1_num)
    class_weight_list += comput_weight(layer_2_num)
    class_weight_list += comput_weight(layer_3_num)
    class_weight_list += comput_weight(layer_4_num)
    # class_weight_list += comput_weight(layer_4_num, type='custom', degree=config.layer_4_weight_degree)

    return torch.tensor(class_weight_list, dtype=torch.float32)

def comput_weight(label_num, type='balanced', degree=0.75):
    all_label = []
    for key, value in label_num.items():
        for i in range(value):
            all_label.append(key)
    classes = np.unique(all_label)

    # 得到的是平衡之后的weight
    if type == 'balanced':
        weight = compute_class_weight(class_weight=type, classes=classes, y=all_label)

    elif type == 'custom':
        balanced_degree = degree  # 平衡程度，以0.75的程度靠近均值
        average_num = int(len(all_label)/len(label_num))
        weight = []
        for key, value in label_num.items():
            if average_num >= value:
                should_num = (average_num - value) * balanced_degree + value
            else:
                should_num = value - (value - average_num) * balanced_degree
            weight.append(should_num/value)
        weight = numpy.array(weight)

    return weight.tolist()

def process_csv_data(path, config, label_map, char_map):
    enz_data = pd.read_csv(path)
    sequence = enz_data.iloc[:, 2:3].values 
    main = enz_data.iloc[:, 5:6].values 
    child1 = enz_data.iloc[:, 6:7].values  
    child2 = enz_data.iloc[:, 7:8].values 
    length = enz_data.iloc[:, 3:4].values  

    # 提取字符串内容和标签
    sequence_list, main_list, child1_list, child2_list = [], [], [], []

    for idx in range(len(sequence)):
        sequence_list.append(str(sequence[idx][0]).replace('\n', '').replace(' ', ''))
        main_list.append(str(main[idx][0]).replace('\n', '').replace(' ', '').replace('EC', ''))
        # child1_list.append(str(child1[idx][0]).replace('\n', '').replace(' ', ''))
        # child2_list.append(str(child2[idx][0]).replace('\n', '').replace(' ', ''))

    seq_id_list, single_id_list, multi_id_list, nof_id_list, length_list = [], [], [], [], []
    # 先处理序列变成id先
    for i in range(len(sequence_list)):
        seq_temp_id = []
        char_list = list(sequence_list[i].strip())  # 字符列表
        seq_len = len(char_list)
        # 统一长度
        if seq_len < config.seq_len:
            char_list.extend([config.PAD] * (config.seq_len - seq_len))
        else:
            char_list = char_list[:config.seq_len]
        # 转换成id
        id_list = []
        for char in char_list:
            seq_temp_id.append(char_map.get(char, char_map.get(config.PAD)))
        seq_id_list.append(seq_temp_id)

        len_temp = int(str(length[i][0]).replace('\n', '').replace(' ', ''))
        if len_temp > config.seq_len:
            length_list.append(config.seq_len)
        else:
            length_list.append(len_temp)

    # 计算纯属亚子类的类别有多少个
    only_child2_num = len(label_map)
    for key, value in label_map.items():
        num = str(key).replace('\n', '').replace(' ', '')
        num_list = num.split('.')
        if len(num_list) == 3:
            only_child2_num -= 1
    # 处理single_id_list, multi_id_list, nof_id_list
    for i in range(len(main_list)):
        single_id_list.append(label_map[main_list[i]])

        # multi_temp_id = []
        # multi_temp_id.append(label_map[main_list[i]])
        # multi_temp_id.append(label_map[child1_list[i]])
        # multi_temp_id.append(label_map[child2_list[i]])
        # multi_id_list.append(multi_temp_id)
        #
        # nof_id_list.append(label_map[child2_list[i]] - only_child2_num)

    return seq_id_list, length_list, single_id_list

def process_biovec_data(path, config, label_map, type):
    enz_data = pd.read_csv(path)
    ec_column = enz_data.iloc[:, 2:3].values  # 获取所有EC号

    multi_label_onehot = []  # 需要返回的是one-hot格式
    multi_label = []  # 非onehot格式
    layer_1_label, layer_2_label, layer_3_label = [], [], []
    layer_1_num, layer_2_num, layer_3_num = 0, 0, 0
    for key, value in label_map.items():
        hiera = len(str(key).split('.'))
        if hiera == 1:
            layer_1_num += 1
        if hiera == 2:
            layer_2_num += 1
        if hiera == 3:
            layer_3_num += 1

    for idx in range(len(ec_column)):
        layer_ec_list = str(ec_column[idx][0]).replace('\n', '').replace('EC', '').replace(' ', '').split('.')
        p_main = layer_ec_list[0]
        p_child1 = layer_ec_list[0] + '.' + layer_ec_list[1]
        p_child2 = layer_ec_list[0] + '.' + layer_ec_list[1] + '.' + layer_ec_list[2]
        lst0 = [0] * len(label_map)
        lst0[label_map[p_main]] = 1
        lst0[label_map[p_child1]] = 1
        lst0[label_map[p_child2]] = 1
        multi_label_onehot.append(lst0)

        temp_label = []
        temp_label.append(label_map[p_main])
        temp_label.append(label_map[p_child1])
        temp_label.append(label_map[p_child2])
        multi_label.append(temp_label)

        layer_1_label.append(label_map[p_main])
        layer_2_label.append(label_map[p_child1] - layer_1_num)
        layer_3_label.append(label_map[p_child2] - layer_1_num - layer_2_num)


    with open(config.data_path + type + '_x.pickle', 'rb') as infile:
        seq_biovec = pickle.load(infile, encoding='bytes')

    all_label = {'one_hot': multi_label_onehot,
                 'three_label': multi_label,
                 'layer_1': layer_1_label,
                 'layer_2': layer_2_label,
                 'layer_3': layer_3_label}
    if type == 'eval':
        all_label = {'one_hot': multi_label_onehot[:1000],
                     'three_label': multi_label[:1000],
                     'layer_1': layer_1_label[:1000],
                     'layer_2': layer_2_label[:1000],
                     'layer_3': layer_3_label[:1000]}
        return seq_biovec[:1000], all_label
    else:
        return seq_biovec, all_label

def process_esm_data(path, config, label_map, esm_data_path):
    enz_data = pd.read_csv(path)
    ec_column = enz_data.iloc[:, 2:3].values  # 获取所有EC号

    multi_label_onehot = []  # 需要返回的是one-hot格式
    multi_label = []  # 非onehot格式
    layer_1_label, layer_2_label, layer_3_label, layer_4_label = [], [], [], []
    layer_1_num, layer_2_num, layer_3_num, layer_4_num = 0, 0, 0, 0
    for key, value in label_map.items():
        hiera = len(str(key).split('.'))
        if hiera == 1:
            layer_1_num += 1
        if hiera == 2:
            layer_2_num += 1
        if hiera == 3:
            layer_3_num += 1
        if hiera == 4:
            layer_4_num += 1

    for idx in range(len(ec_column)):
        layer_ec_list = str(ec_column[idx][0]).replace('\n', '').replace('EC', '').replace(' ', '').split('.')
        p_main = layer_ec_list[0]
        p_child1 = layer_ec_list[0] + '.' + layer_ec_list[1]
        p_child2 = layer_ec_list[0] + '.' + layer_ec_list[1] + '.' + layer_ec_list[2]
        p_child3 = layer_ec_list[0] + '.' + layer_ec_list[1] + '.' + layer_ec_list[2] + '.' + layer_ec_list[3]
        # 制作one-hot
        lst0 = [0] * len(label_map)
        lst0[label_map[p_main]] = 1
        lst0[label_map[p_child1]] = 1
        lst0[label_map[p_child2]] = 1
        lst0[label_map[p_child3]] = 1
        multi_label_onehot.append(lst0)

        temp_label = []
        temp_label.append(label_map[p_main])
        temp_label.append(label_map[p_child1])
        temp_label.append(label_map[p_child2])
        temp_label.append(label_map[p_child3])
        multi_label.append(temp_label)

        layer_1_label.append(label_map[p_main])
        layer_2_label.append(label_map[p_child1] - layer_1_num)
        layer_3_label.append(label_map[p_child2] - layer_1_num - layer_2_num)
        layer_4_label.append(label_map[p_child3] - layer_1_num - layer_2_num - layer_3_num)


    # 加载嵌入数据
    # esm_tensor = torch.load(config.data_path + 'esm_f_' + type + '.pt')  # tensor类型的数据
    esm_tensor = torch.load(esm_data_path)  # tensor类型的数据


    all_label = {'one_hot': multi_label_onehot,
                 'three_label': multi_label,
                 'layer_1': layer_1_label,
                 'layer_2': layer_2_label,
                 'layer_3': layer_3_label,
                 'layer_4': layer_4_label}

    return esm_tensor, all_label

# 用来针对price数据集以及其他数据集
def process_other_esm_data(path, esm_data_path):
    enz_data = pd.read_csv(path)
    ec_column = enz_data.iloc[:, 2:3].values  # 获取所有EC号
    name_column = enz_data.iloc[:, 0:1].values  # 获取所有EC号
    layer_ec_list, name_list = [], []

    for idx in range(len(ec_column)):
        layer_ec = ec_column[idx][0].replace('\n', '').replace('EC', '').replace(' ', '')
        layer_ec_list.append(layer_ec)
        name = name_column[idx][0].replace('\n', '').replace(' ', '')
        name_list.append(name)


    # 加载嵌入数据
    # esm_tensor = torch.load(config.data_path + 'esm_f_' + type + '.pt')  # tensor类型的数据
    esm_tensor = torch.load(esm_data_path)  # tensor类型的数据

    return esm_tensor, layer_ec_list, name_list

# 用来针对k_fold数据集以及其他数据集
def process_kfold_esm_data(path, esm_data_path, kfold_path):
    # 读取文件：
    a = np.load(kfold_path)  # np.array格式
    index_list = []
    index_list = a.tolist()   # 转换成list

    enz_data = pd.read_csv(path)
    ec_column = enz_data.iloc[:, 2:3].values  # 获取所有EC号
    name_column = enz_data.iloc[:, 0:1].values  # 获取所有EC号
    layer_ec_list, name_list = [], []

    # 改成根据文件遍历
    for idx in range(len(index_list)):
        layer_ec = ec_column[index_list[idx]][0].replace('\n', '').replace('EC', '').replace(' ', '')
        layer_ec_list.append(layer_ec)
        name = name_column[index_list[idx]][0].replace('\n', '').replace(' ', '')
        name_list.append(name)


    # 加载嵌入数据
    # esm_tensor = torch.load(config.data_path + 'esm_f_' + type + '.pt')  # tensor类型的数据
    esm_tensor = torch.load(esm_data_path)  # tensor类型的数据

    return esm_tensor[index_list], layer_ec_list, name_list

def get_kfold_dataloader(config, label_map, train_index, test_index):
    esm_tensor, label = \
        process_esm_index_data(config, label_map, train_index)  # 获取数据
    train_data = My_Dataset(esm_tensor, label)
    train_data_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, collate_fn=collate_fn,
                                   shuffle=True, num_workers=2, pin_memory=True)

    esm_tensor, label = \
        process_esm_index_data(config, label_map, test_index)  # 获取数据
    eval_data = My_Dataset(esm_tensor, label)
    eval_data_loader = DataLoader(dataset=eval_data, batch_size=config.batch_size, collate_fn=collate_fn,
                                  shuffle=False, num_workers=2, pin_memory=True)


    return train_data_loader, eval_data_loader

def process_esm_index_data(config, label_map, index):
    enz_data = pd.read_csv(config.kfold_dataset_csv)
    ec_column = enz_data.iloc[:, 2:3].values  # 获取所有EC号

    multi_label_onehot = []  # 需要返回的是one-hot格式
    multi_label = []  # 非onehot格式
    layer_1_label, layer_2_label, layer_3_label, layer_4_label = [], [], [], []
    layer_1_num, layer_2_num, layer_3_num, layer_4_num = 0, 0, 0, 0
    for key, value in label_map.items():
        hiera = len(str(key).split('.'))
        if hiera == 1:
            layer_1_num += 1
        if hiera == 2:
            layer_2_num += 1
        if hiera == 3:
            layer_3_num += 1
        if hiera == 4:
            layer_4_num += 1

    for i in range(len(index)):
        idx = index[i]
        layer_ec_list = str(ec_column[idx][0]).replace('\n', '').replace('EC', '').replace(' ', '').split('.')
        p_main = layer_ec_list[0]
        p_child1 = layer_ec_list[0] + '.' + layer_ec_list[1]
        p_child2 = layer_ec_list[0] + '.' + layer_ec_list[1] + '.' + layer_ec_list[2]
        p_child3 = layer_ec_list[0] + '.' + layer_ec_list[1] + '.' + layer_ec_list[2] + '.' + layer_ec_list[3]
        # 制作one-hot
        lst0 = [0] * len(label_map)
        lst0[label_map[p_main]] = 1
        lst0[label_map[p_child1]] = 1
        lst0[label_map[p_child2]] = 1
        lst0[label_map[p_child3]] = 1
        multi_label_onehot.append(lst0)

        temp_label = []
        temp_label.append(label_map[p_main])
        temp_label.append(label_map[p_child1])
        temp_label.append(label_map[p_child2])
        temp_label.append(label_map[p_child3])
        multi_label.append(temp_label)

        layer_1_label.append(label_map[p_main])
        layer_2_label.append(label_map[p_child1] - layer_1_num)
        layer_3_label.append(label_map[p_child2] - layer_1_num - layer_2_num)
        layer_4_label.append(label_map[p_child3] - layer_1_num - layer_2_num - layer_3_num)


    # 加载嵌入数据
    esm_tensor = torch.load(config.kfold_dataset_esm)  # tensor类型的数据
    esm_tensor = esm_tensor[index]


    all_label = {'one_hot': multi_label_onehot,
                 'three_label': multi_label,
                 'layer_1': layer_1_label,
                 'layer_2': layer_2_label,
                 'layer_3': layer_3_label,
                 'layer_4': layer_4_label}

    return esm_tensor, all_label

def get_bio_index_data(config, label_map, index):
    enz_data = pd.read_csv(config.creat_vocab_data_path)
    ec_column = enz_data.iloc[:, 2:3].values  # 获取所有EC号

    multi_label_onehot = []  # 需要返回的是one-hot格式
    multi_label = []  # 非onehot格式
    layer_1_label, layer_2_label, layer_3_label = [], [], []
    layer_1_num, layer_2_num, layer_3_num = 0, 0, 0
    for key, value in label_map.items():
        hiera = len(str(key).split('.'))
        if hiera == 1:
            layer_1_num += 1
        if hiera == 2:
            layer_2_num += 1
        if hiera == 3:
            layer_3_num += 1

    for i in range(len(index)):
        idx = index[i]
        layer_ec_list = str(ec_column[idx][0]).replace('\n', '').replace(' ', '').split('.')
        p_main = layer_ec_list[0]
        p_child1 = layer_ec_list[0] + '.' + layer_ec_list[1]
        p_child2 = layer_ec_list[0] + '.' + layer_ec_list[1] + '.' + layer_ec_list[2]
        lst0 = [0] * len(label_map)
        lst0[label_map[p_main]] = 1
        lst0[label_map[p_child1]] = 1
        lst0[label_map[p_child2]] = 1
        multi_label_onehot.append(lst0)

        temp_label = []
        temp_label.append(label_map[p_main])
        temp_label.append(label_map[p_child1])
        temp_label.append(label_map[p_child2])
        multi_label.append(temp_label)

        layer_1_label.append(label_map[p_main])
        layer_2_label.append(label_map[p_child1] - layer_1_num)
        layer_3_label.append(label_map[p_child2] - layer_1_num - layer_2_num)


    with open(config.data_path  + 'data_x.pickle', 'rb') as infile:
        seq_biovec = pickle.load(infile, encoding='bytes')
    seq_biovec = seq_biovec[index]

    all_label = {'one_hot': multi_label_onehot,
                 'three_label': multi_label,
                 'layer_1': layer_1_label,
                 'layer_2': layer_2_label,
                 'layer_3': layer_3_label}
    return seq_biovec, all_label

def kfold_get_y(config, label_map):
    y = []
    # 统计第四层的
    enz_data = pd.read_csv(config.kfold_dataset_csv)
    ec_column = enz_data.iloc[:, 2:3].values  # 获取所有EC号

    for i in range(len(ec_column)):
        ec_list = ec_column[i][0].replace('\n', '').replace(' ', '').split('.')
        layer_4_ec = ec_list[0] + '.' + ec_list[1] + '.' + ec_list[2] + '.' + ec_list[3]
        y.append(label_map[layer_4_ec])
    return y

def collate_fn(batch):

    esm_tensor_list, multi_label_onehot_list, multi_label_list, layer_1_list, \
    layer_2_list, layer_3_list, layer_4_list = [], [], [], [], [], [], []

    for i, (esm, multi_label_onehot, multi_label, layer_1, layer_2, layer_3, layer_4) in enumerate(batch):
        esm_tensor_list.append(esm)
        multi_label_onehot_list.append(multi_label_onehot)
        multi_label_list.append(multi_label)
        layer_1_list.append(layer_1)
        layer_2_list.append(layer_2)
        layer_3_list.append(layer_3)
        layer_4_list.append(layer_4)

    # biovec_numpy = numpy.array(biovec_list)
    # multi_label_id_numpy = numpy.array(multi_label_onehot_list)
    esm_tensor = torch.stack(esm_tensor_list, 0)  # torch.stack list tensor 堆叠
    multi_label_onehot_tensor = torch.tensor(numpy.array(multi_label_onehot_list), dtype=torch.float32)
    layer_1_list_tensor = torch.tensor(numpy.array(layer_1_list), dtype=torch.long)
    layer_3_list_tensor = torch.tensor(numpy.array(layer_3_list), dtype=torch.long)

    all_label = {'one_hot': multi_label_onehot_tensor,
                 'three_label': multi_label_list, 
                 'layer_1': layer_1_list,
                 'layer_2': layer_2_list,
                 'layer_3': layer_3_list,
                 'layer_4': layer_4_list}

    return (esm_tensor, all_label)

def other_collate_fn(batch):

    esm_tensor_list, str_ec_list, name_list = [], [], []

    for i, (esm, str_ec, name) in enumerate(batch):
        esm_tensor_list.append(esm)
        str_ec_list.append(str_ec)
        name_list.append(name)


    # biovec_numpy = numpy.array(biovec_list)
    # multi_label_id_numpy = numpy.array(multi_label_onehot_list)
    esm_tensor = torch.stack(esm_tensor_list, 0)  # torch.stack list tensor 堆叠

    return (esm_tensor, str_ec_list, name_list)


