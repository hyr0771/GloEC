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
from tqdm import tqdm

# 屏蔽警告
import warnings
warnings.filterwarnings("ignore")

from loguru import logger


# from Transformer import *

class ClassificationTrainer(object):
    def __init__(self, label_map, config, loss_fn):
        self.label_map = label_map  # 词频map
        self.config = config
        self.criterion = loss_fn
        if self.config.iskfold:
            self.label_length = config.similarity_half_label_length  # 各层类别数
        else:
            self.label_length = config.label_length  # 各层类别数
        # with open(config.hiera_json, 'r+') as file:
        #     self.hierar_relations = json.loads(file.read())  # 将json格式文件转化为python的字典文件

    def train(self, data_loader, model, optimizer):
        # for param_group in optimizer.param_groups[:2]:
        #     param_group["lr"] = lr
        model.train()
        return self.run(data_loader, model, optimizer, is_train=True)

    def eval(self, data_loader, model, optimizer):
        model.eval()
        return self.run(data_loader, model, optimizer, is_train=False)

    def run(self, data_loader, model, optimizer, is_train):
        total_loss = 0.
        predict_probs = []
        target_labels = []
        layer_1_label, layer_2_label, layer_3_label, layer_4_label = [], [], [], []
        num_batch = data_loader.__len__()

        if is_train:
            loop = tqdm(data_loader, desc='Train')
        else:
            loop = tqdm(data_loader, desc='Eval')

        for batch_esm, batch_label in loop:
            input_data = batch_esm.to(self.config.device).clone().detach()
            optimizer.zero_grad()  # 改
            logits = model(input_data)

            if self.config.use_hierar_penalty:
                recursive_params = model.classifier.weight
            else:
                recursive_params = None

            loss = self.criterion(logits, batch_label['one_hot'].to(self.config.device), recursive_params)
            # predict_results = torch.softmax(logits, dim=1)
            # predict_probs.extend(predict_results.tolist())
            # 单分类
            # predict_tensor = torch.argmax(predict_results, 1)
            # predict_probs += predict_tensor.tolist()
            # 多分类
            if not is_train:
                predict_results = torch.sigmoid(logits).cpu().tolist()
                predict_probs.extend(predict_results)
                # target_labels.extend(batch_label['three_label'])
                layer_1_label += (batch_label['layer_1'])
                layer_2_label += (batch_label['layer_2'])
                layer_3_label += (batch_label['layer_3'])
                layer_4_label += (batch_label['layer_4'])

            total_loss += loss.item()

            if is_train:
                loss.backward()
                optimizer.step()
                continue

            # 边打印边输出信息
            # loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
            # loop.set_postfix(loss=running_loss, acc=running_acc)
        if is_train:
            return total_loss/num_batch

        else:
            perform_dict = layer_evaluate(predict_probs, layer_1_label, layer_2_label, layer_3_label, layer_4_label,
                                     self.label_length)
            return total_loss / num_batch, perform_dict

def train(config, now_time):
    label_map = get_label_map(config)
    # for key, value in label_map.items():
    #     print(key)
    class_weight = get_weight(config)
    train_data_loader, validate_data_loader = \
        get_type_dataloader(config, label_map, type='train')
    # test_data_loader = get_type_dataloader(config, label_map, type='test')

    model = get_model(config, label_map, class_num=len(label_map))

    loss_fn = ClassificationLoss(config, label_map, class_weight)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.landa)
    lr_controller = Lr_contronller(config, now_time, tolerate=7)
    trainer = ClassificationTrainer(label_map, config, loss_fn)


    for epoch in range(1, config.epoch + 1):
        start_time = time.time()
        train_loss = trainer.train(train_data_loader, model, optimizer)

        # 训练完一次就验证集跑一次
        with torch.no_grad():
            eval_loss, eval_perform_dict = trainer.eval(validate_data_loader, model, optimizer)  # 返回F1分数
        time_used = time.time() - start_time

        logger.info('epoch%d -train: loss:%.4f   time:%.1fs' % (epoch, train_loss, time_used))
        logger.info('-eval: layer1_f1:%.3f   layer2_f1:%.3f   layer3_f1:%.3f   layer4_f1:%.3f' %
              (eval_perform_dict['layer_1']['f1'], eval_perform_dict['layer_2']['f1'], eval_perform_dict['layer_3']['f1'], eval_perform_dict['layer_4']['f1']))
        logger.info('lr = ' + str(scheduler.get_last_lr()))

        lr_controller.look_parameters(eval_perform_dict['layer_1']['f1'], model, scheduler)  # 返回True表示该下降学习率了
        if lr_controller.earl_stop:
            logger.info('earl stop')
            break

def kfold_train(config, now_time):
    skf = StratifiedKFold(n_splits=config.kfold, shuffle=False)  # 按照类别分布一致来分数据集
    label_map = get_label_map(config)
    # 获得整个的数据集
    y = kfold_get_y(config, label_map)
    fold = 0
    for train_index, eval_index in skf.split(y, y):
        fold += 1
        logger.info('strat a new fold :' + str(fold))
        # X_train, X_test = X[train_index], X[test_index]
        # y_train, y_test = y[train_index], y[test_index]
        # 保存一下train_index 和 eval_index
        np.save('../fold_data/fold_' + str(fold) + '_train.npy', np.array(train_index))
        np.save('../fold_data/fold_' + str(fold) + '_eval.npy', np.array(eval_index))
        train_data_loader, eval_data_loader = get_kfold_dataloader(config, label_map, train_index, eval_index)

        model = get_model(config, label_map, class_num=len(label_map))

        loss_fn = ClassificationLoss(config, label_map)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.landa)
        lr_controller = Lr_contronller(config, now_time, tolerate=7)
        # lr_controller.reset_lr()
        trainer = ClassificationTrainer(label_map, config, loss_fn)

        for epoch in range(1, config.kfold_epoch + 1):
            start_time = time.time()
            train_loss = trainer.train(train_data_loader, model, optimizer)

            time_used = time.time() - start_time
            logger.info('epoch%d -train: loss:%.4f   time:%.1fs' % (epoch, train_loss, time_used))

            # 什么时候跑验证集合？
            if epoch >= config.kfold_epoch - 10:
                # 训练完一次就验证集跑一次
                with torch.no_grad():
                    eval_loss, eval_perform_dict = trainer.eval(eval_data_loader, model, optimizer)  # 返回F1分数
                logger.info('-eval: layer1_f1:%.3f   layer2_f1:%.3f   layer3_f1:%.3f   layer4_f1:%.3f' %
                            (eval_perform_dict['layer_1']['f1'], eval_perform_dict['layer_2']['f1'],
                             eval_perform_dict['layer_3']['f1'],
                             eval_perform_dict['layer_4']['f1']))
                logger.info('lr = ' + str(scheduler.get_last_lr()))
                lr_controller.look_parameters(eval_perform_dict['layer_1']['f1'], model, scheduler,
                                              fold=fold)  # 返回True表示该下降学习率了
                if lr_controller.earl_stop:
                    logger.info('earl stop')
                    break


        if not config.sure_full_kfold:
            break

if __name__ == '__main__':
    config = Config()
    # 封装设定固定种子
    set_seed(2023)
    now_time = dt.datetime.now().strftime('%m' + "%d" + '%H' + '%M')  # 用来记录输出保存的模型
    logger.add('../logs/' + str(now_time) + '.log', format="{time} {level} {message}")
    logger.info('model name : ' + config.model_name + '_' + str(now_time))
    logger.info('batch_size: ' + str(config.batch_size))
    logger.info('是否使用GCN: ' + str(config.use_GCN))
    logger.info('层级惩罚 : ' + str(config.use_hierar_penalty))
    logger.info('is Kfold : ' + str(config.iskfold))

    if config.use_GCN:
        logger.info('GCN层数: ' + str(config.gcn_layer))
    logger.info('第四层平衡程度:' + str(config.layer_4_weight_degree))

    if config.is_continue_train:
        logger.info('continue train: ESM_' + str(config.continue_train_num))

    if config.iskfold:
        kfold_train(config, now_time)
    else:
        train(config, now_time)
    # train(config, now_time)
    # 全数据集哪用做交叉验证啦傻逼玩意

