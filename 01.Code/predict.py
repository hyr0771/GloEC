from model_util import *
import numpy as np
import torch
from torchsummary import summary
from HiAGM import HiAGM
import datetime as dt
import os
import random
from model_layer import *
from dataset_util import *
from config_util import Config
from sklearn.model_selection import StratifiedKFold
import bisect
from get_esm import get_esm_data

def predict(config):
    model_index = 'ESM_06131214'
    label_map = get_label_map(config)
    # _, eval_data_loader = get_type_dataloader(config, label_map, type='train')
    # new_data_loader = get_type_dataloader(config, label_map, type='test')
    # price_data_loader = get_type_dataloader(config, label_map, type='price')
    # np_data_loader = get_type_dataloader(config, label_map, type='np')
    # isoform_data_loader = get_type_dataloader(config, label_map, type='isoform')
    # cofactor_data_loader = get_type_dataloader(config, label_map, type='co')
    nc_data_loader = get_type_dataloader(config, label_map, type='nc')
    dataset_name = 'nc_dataset' + '_' + model_index


    model = get_model(config, label_map, class_num=len(label_map))

    model.load_state_dict(torch.load('../Save_model/' + model_index + '.pth', map_location=torch.device('cpu')))

    model.eval()
    with torch.no_grad():
        perform_dict = run(nc_data_loader, model, label_map, config, dataset_name=dataset_name)

    print('test result : ')
    print('layer_1 :   f1:%.4f    precision:%.4f    recall:%.4f'%
          (perform_dict['layer_1']['f1'], perform_dict['layer_1']['precision'], perform_dict['layer_1']['recall']))
    print('layer_2 :   f1:%.4f    precision:%.4f    recall:%.4f'%
          (perform_dict['layer_2']['f1'], perform_dict['layer_2']['precision'], perform_dict['layer_2']['recall']))
    print('layer_3 :   f1:%.4f    precision:%.4f    recall:%.4f'%
          (perform_dict['layer_3']['f1'], perform_dict['layer_3']['precision'], perform_dict['layer_3']['recall']))
    print('layer_4 :   f1:%.4f    precision:%.4f    recall:%.4f'%
          (perform_dict['layer_4']['f1'], perform_dict['layer_4']['precision'], perform_dict['layer_4']['recall']))
    # print('layer1_f1:%.3f   layer2_f1:%.3f   layer3_f1:%.3f   layer4_f1:%.3f'%
    #       (f1_dict['layer_1'], f1_dict['layer_2'], f1_dict['layer_3'], f1_dict['layer_4']))

def run(data_loader, model, label_map, config, dataset_name, split_area=None, label_num_dict=None):
    # split_area label_num_dict 只有在kfold计算时才用到，split_area表示按照什么比例划分，label_num_dict是一个字典，表示每种label在训练集中的个数
    predict_probs = []
    true_label = []
    name_list = []

    for i, (batch_esm, batch_str_label, name) in enumerate(data_loader):
        logits = model(batch_esm)

        predict_results = torch.sigmoid(logits)
        predict_probs.extend(predict_results.tolist())
        true_label += batch_str_label
        name_list += name

    # 获取预测文件
    get_predict_file(label_map, predict_probs, true_label, name_list, config, dataset_name=dataset_name)

    # 根据文件计算指标
    if split_area == None:
        perform_dict = get_other_dataset_perform(label_map,
                                                 predict_file='../Data/predict_result/' + dataset_name + '.csv')
    else:
        perform_dict = get_kfold_dataset_perform(label_map,
                                                 predict_file='../Data/predict_result/' + dataset_name + '.csv',
                                                 split_area=split_area, label_num_dict=label_num_dict)
    # 计算指标
    return perform_dict

def k_fold_predict(config, istest_one=False, issplit_area=False):
    # istest_one 是否自己一个个进行创尝试？
    # 统计第四层阶级先
    kfold_num = 10
    ds = pd.read_csv(config.kfold_dataset_csv)
    name_column = ds.iloc[:, 0:1].values  # 获取所有EC号
    sequence_column = ds.iloc[:, 1:2].values  # 获取所有EC号
    ecnum_column = ds.iloc[:, 2:3].values  # 获取所有EC号

    # 只统计第四层
    label_num_dict = {}
    for i in range(len(name_column)):
        ec = str(ecnum_column[i][0]).replace('\n', '').replace(' ', '')
        if ec not in label_num_dict.keys():
            label_num_dict[ec] = 1
        else:
            label_num_dict[ec] += 1

    print(label_num_dict)
    # 字典排序：
    # 我为什么要排序啊？像个傻逼一样？？？？
    split_area = [0, 5, 10, 20, 50, 100]  # 小0到5叫0列表
    model_index = 'ESM_07041450'
    label_map = get_label_map(config)

    # 自己进行一个个的尝试
    if istest_one:
        test_num = 2
        model_name = model_index + '_' + str(test_num)
        kfold_eval_index_path = '../fold_data/fold_' + str(test_num) + '_eval.npy'
        save_dataset_name = 'fold_' + str(test_num)
        isoform_data_loader = get_type_dataloader(config, label_map, type='kfold',
                                                  kfold_eval_index_path=kfold_eval_index_path)

        model = get_model(config, label_map, class_num=len(label_map))

        model.load_state_dict(torch.load('../Save_model/' + model_name + '.pth', map_location=torch.device('cpu')))
        model.eval()
        with torch.no_grad():
            split_perform_dict = run(isoform_data_loader, model, label_map, config, dataset_name=save_dataset_name,
                                     split_area=None, label_num_dict=None)

        print('kfold ' + str(test_num) + '  result : ')
        for key, value in split_perform_dict.items():
            print(' 小于 ' + key)
            print('f1:%.4f    precision:%.4f    recall:%.4f' % (value['f1'], value['precision'], value['recall']))

    else:
        all_fold_perform_dict = {}
        for i in range(1, kfold_num + 1):
            print(' compute the ' + str(i) + ' fold perform......')
            test_num = i
            model_name = model_index + '_' + str(test_num)
            kfold_eval_index_path = '../fold_data/fold_' + str(test_num) + '_eval.npy'
            save_dataset_name = 'fold_' + str(test_num)
            isoform_data_loader = get_type_dataloader(config, label_map, type='kfold',
                                                      kfold_eval_index_path=kfold_eval_index_path)

            model = get_model(config, label_map, class_num=len(label_map))

            model.load_state_dict(torch.load('../Save_model/' + model_name + '.pth', map_location=torch.device('cpu')))
            model.eval()
            with torch.no_grad():
                if issplit_area:
                    split_perform_dict = run(isoform_data_loader, model, label_map, config,
                                             dataset_name=save_dataset_name,
                                             split_area=split_area, label_num_dict=label_num_dict)
                else:
                    split_perform_dict = run(isoform_data_loader, model, label_map, config,
                                             dataset_name=save_dataset_name)

            all_fold_perform_dict[str(test_num)] = split_perform_dict

        print('all_fold avarager performation : ')
        average_result_dict = {}

        if issplit_area:
            for fold, perform_dict in all_fold_perform_dict.items():
                for split_area, f1_pre_recall_dict in perform_dict.items():
                    if split_area not in average_result_dict.keys():
                        average_result_dict[split_area] = {}
                        average_result_dict[split_area]['f1'] = 0.0
                        average_result_dict[split_area]['precision'] = 0.0
                        average_result_dict[split_area]['recall'] = 0.0
                    average_result_dict[split_area]['f1'] += f1_pre_recall_dict['f1'] / kfold_num
                    average_result_dict[split_area]['precision'] += f1_pre_recall_dict['precision'] / kfold_num
                    average_result_dict[split_area]['recall'] += f1_pre_recall_dict['recall'] / kfold_num

            # 打印平均值结果
            for key, value in average_result_dict.items():
                print(' 小于 ' + key)
                print('f1:%.4f    precision:%.4f    recall:%.4f' % (value['f1'], value['precision'], value['recall']))

        else:
            for fold, perform_dict in all_fold_perform_dict.items():
                for layer, f1_pre_recall_dict in perform_dict.items():
                    if layer not in average_result_dict.keys():
                        average_result_dict[layer] = {}
                        average_result_dict[layer]['f1'] = 0.0
                        average_result_dict[layer]['precision'] = 0.0
                        average_result_dict[layer]['recall'] = 0.0
                    average_result_dict[layer]['f1'] += f1_pre_recall_dict['f1'] / kfold_num
                    average_result_dict[layer]['precision'] += f1_pre_recall_dict['precision'] / kfold_num
                    average_result_dict[layer]['recall'] += f1_pre_recall_dict['recall'] / kfold_num

            # 打印平均值结果
            for key, value in average_result_dict.items():
                print(key + ':')
                print('f1:%.4f    precision:%.4f    recall:%.4f' % (value['f1'], value['precision'], value['recall']))


def isoform_single_predict():
    # 我真是傻逼
    model_index = 'ESM_06131214'
    isoform_name = 'P26361-3'
    label_map = get_label_map(config)
    # 获得你要测试的isoform的pt,怎么获得？跨文件查找
    isoform_esm_pt = get_esm_data(isoform_name)
    model = get_model(config, label_map, class_num=len(label_map))
    model.load_state_dict(torch.load('../Save_model/' + model_index + '.pth', map_location=torch.device('cpu')))

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

    model.eval()
    with torch.no_grad():
        logits = model(isoform_esm_pt)
        predict_results = torch.sigmoid(logits)

    label_length = [layer_1_class, layer_2_class, layer_3_class, layer_4_class]
    layer = torch.split(predict_results, label_length, dim=1)

    # 获得最大值处
    layer_4_predict_num = int(torch.argmax(layer[3].data, 1))
    p_child3 = (id_label_map[layer_4_predict_num + layer_1_class + layer_2_class + layer_3_class])
    print(isoform_name + ' predict result is : ' + str(p_child3))


if __name__ == '__main__':
    config = Config()
    # 封装设定固定种子
    # set_seed(2023)
    # print('batch_size: ' + str(config.batch_size))
    print('use GCN : ' + str(config.use_GCN))
    # k_fold_predict(config, istest_one=False, issplit_area=False)
    # predict(config)
    isoform_single_predict()
