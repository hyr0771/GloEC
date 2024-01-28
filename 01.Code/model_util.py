import numpy as np
import torch
from torchsummary import summary
from HiAGM import HiAGM
from ESM import ESM_Model
import datetime as dt
import os
import pandas as pd
import random
from sklearn.metrics import precision_score, recall_score, f1_score  # 不一定要用
from loguru import logger
import bisect

def get_optimizer(config, model):
    params = model.get_parameter_optimizer_dict()
    if config.optimizer_type == 'Adam':
        return torch.optim.Adam(lr=config.learning_rate,
                                params=params)

def init_tensor(tensor, init_type='uniform', low=0, high=1):
        """Init torch.Tensor
        Args:
            tensor: Tensor to be initialized.
            init_type: Init type, candidate can be found in InitType.
            low: The lower bound of the uniform distribution,
                useful when init_type is uniform.
            high: The upper bound of the uniform distribution,
                useful when init_type is uniform.
            mean: The mean of the normal distribution,
                useful when init_type is normal.
            std: The standard deviation of the normal distribution,
                useful when init_type is normal.
            activation_type: For xavier and kaiming init,
                coefficient is calculate according the activation_type.
            fan_mode: For kaiming init, fan mode is needed
            negative_slope: For kaiming init,
                coefficient is calculate according the negative_slope.
        Returns:
        """
        if init_type == 'uniform':
            return torch.nn.init.uniform_(tensor, a=low, b=high)

def get_model(config, label_map, class_num=7):

    if config.model_name == 'Transformer':
        # model = Transformer(config, char_map, label_map)
        # # 设置变更模型参数 # map_location='cpu' 在本机器上训练测试时要加上,免得报错
        # if config.is_continue_train:
        #     if config.device == 'cpu':
        #         pretrained_dict = torch.load(config.model_save_path + config.model_type + '/' + config.model_name +
        #                                      '/train_' +config.contin_num + '.pth', map_location='cpu')
        #     else:
        #         pretrained_dict = torch.load(
        #             config.model_save_path + config.model_type + '/' +
        #             config.model_name + '/train_' + config.contin_num + '.pth')
        #     model.load_state_dict(pretrained_dict, strict=False)  # strict=False 表示如果遇到名字不同的层则跳过
        return '上面全部取消注释'

    elif config.model_name == 'CDIL_cnn':
        # model = CDIL_CNN(len(char_map), config.embed_dim, class_num, config.CDIL_cnn.HIDDEN_CHANNEL)
        return '把上面的取消注释'
        # 设置变更模型参数 # map_location='cpu' 在本机器上训练测试时要加上,免得报错

    elif config.model_name == 'HiAGM':
        model = HiAGM(config, label_map)

    elif config.model_name == 'ESM':
        model = ESM_Model(config, label_map)

    if config.is_continue_train:

        model.load_state_dict(torch.load('../Save_model/ESM_' + config.continue_train_num + '.pth', map_location=torch.device('cpu')))

    summary(model)
    model = model.to(config.device)
    return model

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU随机种子确定
    torch.cuda.manual_seed(seed)  # GPU随机种子确定
    torch.cuda.manual_seed_all(seed)  # 所有的GPU设置种子
    torch.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭
    torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法

def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f

def evaluate(epoch_predicts, epoch_labels, label_map, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'

    # epoch_gold_label = list()
    # # get id label name of ground truth
    # for sample_labels in epoch_labels:
    #     sample_gold = []
    #     for label in sample_labels:
    #         assert label in id2label.keys(), print(label)
    #         sample_gold.append(id2label[label])
    #     epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    # list of list ,长宽分别为类别个数
    confusion_count_list = [[0 for _ in range(len(label_map.keys()))] for _ in range(len(label_map.keys()))]
    right_count_list = [0 for _ in range(len(label_map.keys()))]  # 长度为类别数的全0list
    gold_count_list = [0 for _ in range(len(label_map.keys()))]
    predicted_count_list = [0 for _ in range(len(label_map.keys()))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        # sample_predict_label_list = [id2label[i] for i in sample_predict_id_list]

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for key, value in label_map.items():
        precision_dict[key], recall_dict[key], fscore_dict[key] = _precision_recall_f1(right_count_list[value],
                                                                                             predicted_count_list[value],
                                                                                             gold_count_list[value])
        right_total += right_count_list[value]
        gold_total += gold_count_list[value]
        predict_total += predicted_count_list[value]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1}

def layer_evaluate(predict_batch_list, layer_1_label, layer_2_label, layer_3_label, layer_4_label, label_length):
    predict_tensor = torch.tensor(predict_batch_list)
    layer = torch.split(predict_tensor, label_length, dim=1)
    layer_1_predict = torch.argmax(layer[0].data, 1).tolist()
    layer_2_predict = torch.argmax(layer[1].data, 1).tolist()
    layer_3_predict = torch.argmax(layer[2].data, 1).tolist()
    layer_4_predict = torch.argmax(layer[3].data, 1).tolist()

    # 返回的是f1 precision and recall
    layer_1_perform = calculate_F1(layer_1_label, layer_1_predict, type='macro')
    layer_2_perform = calculate_F1(layer_2_label, layer_2_predict, type='macro')
    layer_3_perform = calculate_F1(layer_3_label, layer_3_predict, type='macro')
    layer_4_perform = calculate_F1(layer_4_label, layer_4_predict, type='macro')

    # 返回字典套字典吧
    perform_dict = {'layer_1': layer_1_perform, 'layer_2': layer_2_perform,
               'layer_3': layer_3_perform, 'layer_4': layer_4_perform}

    return perform_dict

def get_other_dataset_perform(label_map, predict_file):
    enz_data = pd.read_csv(predict_file)
    true_ec_column = enz_data.iloc[:, 1:2].values  # 获取所有EC号
    child1_column = enz_data.iloc[:, 2:3].values  # 获取所有EC号
    child2_column = enz_data.iloc[:, 3:4].values  # 获取所有EC号
    child3_column = enz_data.iloc[:, 4:5].values  # 获取所有EC号
    child4_column = enz_data.iloc[:, 5:6].values  # 获取所有EC号
    layer_1_label, layer_1_predict = [], []
    layer_2_label, layer_2_predict = [], []
    layer_3_label, layer_3_predict = [], []
    layer_4_label, layer_4_predict = [], []

    layer_1_class, layer_2_class, layer_3_class, layer_4_class = 0, 0, 0, 0
    layer_1_dic, layer_2_dic, layer_3_dic, layer_4_dic = {}, {}, {}, {}
    for key, value in label_map.items():
        hiera = len(str(key).split('.'))
        if hiera == 1:
            layer_1_dic[key] = layer_1_class
            layer_1_class += 1
        if hiera == 2:
            layer_2_dic[key] = layer_2_class
            layer_2_class += 1
        if hiera == 3:
            layer_3_dic[key] = layer_3_class
            layer_3_class += 1
        if hiera == 4:
            layer_4_dic[key] = layer_4_class
            layer_4_class += 1


    for i in range(len(true_ec_column)):
        str_true_label = true_ec_column[i][0].replace(' ', '').replace('\n', '')
        true_label_list = str_true_label.split('.')
        true_1 = true_label_list[0]
        true_2 = true_label_list[0] + '.' + true_label_list[1]
        true_3 = true_label_list[0] + '.' + true_label_list[1] + '.' + true_label_list[2]
        true_4 = true_label_list[0] + '.' + true_label_list[1] + '.' + true_label_list[2] + '.' + true_label_list[3]

        pre_1 = str(child1_column[i][0]).replace(' ', '').replace('\n', '')
        pre_2 = str(child2_column[i][0]).replace(' ', '').replace('\n', '')
        pre_3 = child3_column[i][0].replace(' ', '').replace('\n', '')
        pre_4 = child4_column[i][0].replace(' ', '').replace('\n', '')

        #1
        layer_1_label.append(layer_1_dic[true_1])
        layer_1_predict.append(layer_1_dic[pre_1])

        #2
        if true_2 not in layer_2_dic.keys():
            # continue
            print('警告：label_map不存在第2级真实标签' + true_2 + '   已经忽略')
            # layer_2_dic[true_2] = len(layer_2_dic)
            continue
        layer_2_label.append(layer_2_dic[true_2])
        layer_2_predict.append(layer_2_dic[pre_2])

        # 3
        if true_3 not in layer_3_dic.keys():
            # continue
            print('警告：label_map不存在第3级真实标签' + true_3 + '   已经忽略')
            # layer_3_dic[true_3] = len(layer_3_dic)
            continue
        layer_3_label.append(layer_3_dic[true_3])
        layer_3_predict.append(layer_3_dic[pre_3])

        # 4
        if true_4 not in layer_4_dic.keys():
            # continue
            print('警告：label_map不存在第4级真实标签' + true_4 + '   已经忽略')
            # layer_4_dic[true_4] = len(layer_4_dic)
            continue
        layer_4_label.append(layer_4_dic[true_4])
        layer_4_predict.append(layer_4_dic[pre_4])


    # 返回的是f1 precision and recall
    layer_1_perform = calculate_F1(layer_1_label, layer_1_predict, type='macro')
    layer_2_perform = calculate_F1(layer_2_label, layer_2_predict, type='macro')
    layer_3_perform = calculate_F1(layer_3_label, layer_3_predict, type='macro')
    layer_4_perform = calculate_F1(layer_4_label, layer_4_predict, type='macro')

    # 返回字典套字典吧
    perform_dict = {'layer_1': layer_1_perform, 'layer_2': layer_2_perform,
               'layer_3': layer_3_perform, 'layer_4': layer_4_perform}

    return perform_dict

# 只计算kfold数据集。而且只算第四层
def get_kfold_dataset_perform(label_map, predict_file, split_area, label_num_dict):
    enz_data = pd.read_csv(predict_file)
    true_ec_column = enz_data.iloc[:, 1:2].values  # 获取所有EC号
    child4_column = enz_data.iloc[:, 5:6].values  # 获取所有EC号

    layer_4_class = 0
    layer_4_dic = {}
    for key, value in label_map.items():
        hiera = len(str(key).split('.'))
        if hiera == 4:
            layer_4_dic[key] = layer_4_class
            layer_4_class += 1

    split_dict = {}

    for i in range(len(true_ec_column)):
        true_label = true_ec_column[i][0].replace(' ', '').replace('\n', '')
        pre_4 = child4_column[i][0].replace(' ', '').replace('\n', '')

        # 获取训练集中true_laebl个数：
        train_label_num = label_num_dict[true_label]
        # 获得在split_area中的插入位置
        insert_location = bisect.bisect_left(split_area, train_label_num)

        if insert_location > len(split_area) or insert_location < 0:
            print('出错')
            return
        # 判断这个位置是否已经存在了用来存储了列表
        save_list_true_label_name = str(insert_location) + '_true_label'
        save_list_pre_label_name = str(insert_location) + '_pre_label'
        if save_list_true_label_name not in split_dict.keys() and save_list_pre_label_name not in split_dict.keys():
            split_dict[save_list_true_label_name] = []
            split_dict[save_list_pre_label_name] = []
        split_dict[save_list_true_label_name].append(layer_4_dic[true_label])
        split_dict[save_list_pre_label_name].append(layer_4_dic[pre_4])

    # 根据字典计算指标
    split_perform_dict = {}
    for i in range(int(len(split_dict)/2)):
        index = i + 1
        split_true_label_list = split_dict[str(index) + '_true_label']
        split_pre_label_list = split_dict[str(index) + '_pre_label']
        # 返回值包含f1 percision recall
        split_perform = calculate_F1(split_true_label_list, split_pre_label_list, type='macro')
        split_name = ''
        if int(index) == len(split_area):
            split_name = 'more'
        else:
            split_name = str(split_area[int(index)])
        split_perform_dict[split_name] = split_perform

    return split_perform_dict

def get_predict_file(label_map, predict_batch_list, true_label, name_list, config, dataset_name):
    # label_map to id_label_map
    id_label_map = {}
    for key, value in label_map.items():
        id_label_map[value] = key

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

    label_length = [layer_1_class, layer_2_class, layer_3_class, layer_4_class]

    predict_tensor = torch.tensor(predict_batch_list)
    layer = torch.split(predict_tensor, label_length, dim=1)
    layer_1_predict = torch.argmax(layer[0].data, 1).tolist()
    layer_2_predict = torch.argmax(layer[1].data, 1).tolist()
    layer_3_predict = torch.argmax(layer[2].data, 1).tolist()
    layer_4_predict = torch.argmax(layer[3].data, 1).tolist()

    name, true_EC, p_main, p_child1, p_child2, p_child3 = [], [], [], [], [], []

    for i in range(len(layer_1_predict)):
        p_main.append(id_label_map[layer_1_predict[i]])
        p_child1.append(id_label_map[layer_2_predict[i] + layer_1_class])
        p_child2.append(id_label_map[layer_3_predict[i] + layer_1_class + layer_2_class])
        p_child3.append(id_label_map[layer_4_predict[i] + layer_1_class + layer_2_class + layer_3_class])

        true_EC.append(true_label[i])
        name.append(name_list[i])

    # 保存测试集
    result_csv = pd.DataFrame({'Entry': name, 'True EC': true_EC, 'Predict main EC': p_main,
                         'Predict child1': p_child1, 'Predict child2': p_child2, 'Predict child3': p_child3})
    result_csv.to_csv(config.predict_result_path + dataset_name + '.csv', index=False, sep=',')
    print('保存预测结果到 ：' + config.predict_result_path + dataset_name + '.csv')

def main_class_evaluate(predict_batch_list, layer_1_label):

    f1_dict = calculate_F1(layer_1_label, predict_batch_list, type='macro')

    return f1_dict

def calculate_F1(label, predict, type='macro'):
    f1 = f1_score(label, predict, average=type)
    precision = precision_score(label, predict, average=type)
    recall = recall_score(label, predict, average=type)

    performer = {'f1': f1, 'precision': precision, 'recall': recall}
    return performer


# 学习率指标观察器
class Lr_contronller():
    def __init__(self, config, now_time, tolerate=10, min_lr=0.00008):
        super(Lr_contronller, self).__init__()
        self.best_perform = 0.
        self.tolerate = tolerate
        self.current_tol = 0
        self.model_save = config.model_save_path + config.model_name
        # 获取当前时间
        self.now_time = now_time
        self.min_lr = min_lr
        self.earl_stop = False


    def look_parameters(self, perform, model, scheduler, fold=None):
        if self.best_perform < perform:
            self.best_perform = perform
            self.current_tol = 0

            if fold != None:
                logger.info('saving current model to ' + self.model_save + '_' + str(self.now_time)
                    + '_' + str(fold) + '.pth')
                torch.save(model.state_dict(), self.model_save + '_' + str(self.now_time) + '_' + str(fold) + '.pth')
            else:
                logger.info('saving current model to ' + self.model_save + '_' + str(self.now_time) + '.pth')
                torch.save(model.state_dict(), self.model_save + '_' + str(self.now_time) + '.pth')
        else:
            self.current_tol += 1
            if self.current_tol == self.tolerate:
                self.reduce_lr = True
                self.current_tol = 0
                scheduler.step()
                print('Reduce Lr to ' + str(scheduler.get_last_lr()))
        if scheduler.get_last_lr()[0] <= self.min_lr:
            self.earl_stop = True

    def reset_lr(self):
        self.current_tol = 0
        self.best_perform = 0.

# class lr_control():
#     def __init__(self):


