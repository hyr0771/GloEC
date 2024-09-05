import torch.cuda
from tensorboardX import SummaryWriter
import pandas as pd
from Transformer import *
import numpy as np
import time
import sys
import os
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score  # 不一定要用
import pickle
from LoadData import *
from torch.optim import lr_scheduler

label_map = dict()

def collate_fn(batch):
    target = []
    token_index = []
    # 传入1,2,3级标签
    for i ,(label, sequence) in enumerate(batch):
        #假设dataset返回的是一句句话
        tokens_list = list(sequence.strip())  #字符列表
        seq_len = len(tokens_list)

        # 统一长度
        if seq_len < config.pad_size:
            tokens_list.extend([config.PAD] * (config.pad_size - seq_len))
        else:
            tokens_list = tokens_list[:config.pad_size]

        # 转换成id
        id_list = []
        for word in tokens_list:
            id_list.append(vocab.get(word, vocab.get(config.PAD)))
        # 转换成id
        token_index.append(id_list)   # list to list

        # 处理标签
        label_list = []

        if label == 1:
            target.append(1)
        elif label == 2:
            target.append(2)
        elif label == 3:
            target.append(3)
        elif label == 4:
            target.append(4)
        elif label == 5:
            target.append(5)
        elif label == 6:
            target.append(6)
        elif label == 7:
            target.append(7)
        else:
            target.append(0)

    return (torch.tensor(target).to(torch.int64), torch.tensor(token_index).to(torch.int32))

# 弃用
def train(train_data_loader, eval_data_loader, model, optimizer, num_epoch):
    """此处data_loader是map-style dataset"""
    start_epoch = 0
    start_step = 0
    loss_func = nn.CrossEntropyLoss()

    for epoch_index in range(start_epoch, num_epoch):
        ema_loss = 0   # 指数移动平均loss
        num_batches = len(train_data_loader)
        print('---训练第 ' + str(epoch_index) + ' 个epoch---')
        for batch_index, (target, token_index) in enumerate(train_data_loader):
            optimizer.zero_grad()
            model.train()  # 默认开启？
            step = num_batches * (epoch_index) + batch_index +1
            logits = model(token_index)
            # tag = F.one_hot(target, num_classes=8)
            # print(tag)
            # print(logits)
            # pp = torch.sigmoid(logits)

            bce_loss2 = loss_func(logits, target)   #输出是tensor类型
            ema_loss = 0.9 * ema_loss + 0.1*bce_loss2.item()   #ema_loss本来是int，加上tensor后变成tensor类型,利用item（）将tensor转换成数值，同时防止内存爆炸
            bce_loss2.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            # 保存模型
            # if step % save_step_interval == 0:
            #     os.makedirs(save_path, exist_ok=True)
            #     save_file = os.path.join(save_path,f"step_{step}.pt")
            #     torch.save({
            #         'epoch':epoch_index,
            #         'step':step,
            #         'model_state_dict':model.state_dict(),
            #         'optimizer_state_dict':optimizer.state_dict(),
            #         'loss':ema_loss,
            #     },save_file)
            #     logging.warning(f"checkpoint has been saved in {save_file}")
            # 测试验证集

        #if step % eval_step_interval == 0:
        print('---测试第 ' + str(epoch_index) + ' 个epoch---')
        ema_eval_loss = 0
        total_acc_account = 0
        total_account = 0
        for eval_batch_index, (eval_target, eval_token_index) in enumerate(eval_data_loader):
            model.eval()
            total_account += eval_target.shape[0]
            eval_logits = model(eval_token_index)
            total_acc_account += (torch.argmax(eval_logits, dim=-1) == eval_target).sum().item()
            # print('正确的个数： ' + str(total_acc_account) + '总个数：' + str(total_account))
            eval_bce_loss = loss_func(eval_logits, eval_target)
            ema_eval_loss = 0.9 * ema_eval_loss + 0.1 * eval_bce_loss.item()

def train_epoch(train_data_loader, model, loss_func, optimizer, device):
    train_loss = 0.0
    train_correct = 0
    model.train()
    for i, (target, token_index) in enumerate(train_data_loader):
        # target, token_index = target.to(device), token_index.to(device)
        optimizer.zero_grad()
        target = target.to(device)
        token_index = token_index.to(device)
        output = model(token_index)
        loss = loss_func(output, target)  # 输出是tensor类型
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        train_loss += loss.item() * token_index.size(0)
        predictions = torch.argmax(output.data, 1)  # 统计以行为单位的最大值，计为scores，预测记为predictions
        train_correct += (predictions == target).sum().item()

    return train_loss, train_correct

def valid_epoch(test_data_loader, model, loss_func, device):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    predict_list, labels_list = [], []  # 所有的预测值和标签值
    with torch.no_grad():
        for i, (test_labels, test_token_index) in enumerate(test_data_loader):
            # test_labels, test_token_index = test_labels.to(device), test_token_index.to(device)
            test_labels = test_labels.to(device)
            test_token_index = test_token_index.to(device)
            output = model(test_token_index)
            loss = loss_func(output, test_labels)
            valid_loss += loss.item() * test_token_index.size(0)
            predictions = torch.argmax(output.data, 1)
            val_correct += (predictions == test_labels).sum().item()  # 正确的个数
            predict_list += predictions.tolist()
            labels_list += test_labels.tolist()

        # f1_score()输出的是cpu的,所以最好传入的也是cpu的
        test_labels = test_labels.to('cpu')
        predictions = predictions.to('cpu')
        test_acc = val_correct / len(test_index)
        f1 = f1_score(test_labels, predictions, average='macro')
        precision = precision_score(test_labels, predictions, average='macro')
        recall = recall_score(test_labels, predictions, average='macro')

    print("Test Loss:{:.4f} Test Acc {:.4f}".format(valid_loss, test_acc))
    return test_acc, f1, precision, recall

def process_csv_data(path):
    enz_data = pd.read_csv(path)
    sequence = enz_data.iloc[:, 2:3].values  
    main = enz_data.iloc[:, 5:6].values 
    child1 = enz_data.iloc[:, 6:7].values  
    child2 = enz_data.iloc[:, 7:8].values  

    # 提取字符串内容和标签
    contents = []
    labels1 = []
    labels2 = []
    labels3 = []
    for idx in range(len(sequence)):
        contents.append(str(sequence[idx][0]))
        labels1.append(str(main[idx][0]))
        labels2.append(str(child1[idx][0]))
        labels3.append(str(child2[idx][0]))

    return contents, labels1, labels2, labels3

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    print("使用GPU训练" if torch.cuda.is_available() else "使用CPU训练")

    config = Config()  # 定义所有参数

    # 固定种子
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    writer = SummaryWriter(log_dir='../logs/' + config.data_name)
    print("Loading data...")
    vocab = {}
    if (os.path.exists('../Vocab/' + config.data_name + '.pkl')):
        print('存在' + config.data_name + '.pkl文件， 正在读取')
        with open('../Vocab/' + config.data_name + '.pkl', 'rb') as f:
            vocab = pickle.load(f)
        print(vocab)
    else:
        print('不存在' + config.data_name + '.pkl文件， 开始新建词表')
        vocab = build_vocab(config)  # 构建词表，可能没有用
        # 保存词表
        with open('../Vocab/' + config.data_name + '.pkl', 'wb') as f:
            pickle.dump(vocab, f)

    contents, mian, child1, child2 = process_csv_data(config.data_path)  # 获取数据
    # skf = StratifiedKFold(n_splits=config.k_fold).split(contents, labels)  # 按照标签分布来分配
    skf = StratifiedKFold(n_splits=config.k_fold)  # 按照标签分布来分配

    # 进行10折交叉验证
    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.to(device)  # 需要to(device)的参数只有模型，损失函数和数据集
    foldperf = {}  # 每一折模型的表现
    meanF1, meanPre, meanRe = [], [], []
    for kfold, (train_index, test_index) in enumerate(skf.split(contents, labels)):
        # print(np.array(labels)[test_index]) 
        if kfold >= 1:
            break
        print('Fold {}'.format(kfold+1))
        #模型数据获取
        X_train, X_test = np.array(contents)[train_index], np.array(contents)[test_index]
        y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]
        train_data = My_Dataset(X_train, y_train)
        test_data = My_Dataset(X_test, y_test)
        train_data_loader = DataLoader(dataset=train_data, batch_size=8, collate_fn=collate_fn, shuffle=True)
        test_data_loader = DataLoader(test_data, batch_size=8, collate_fn=collate_fn)

        #模型参数获取
        model = Transformer_model(config, device)  # 这里的device是为model里面的一些变量准备的
        model = model.to(device)
        # print(next(model.parameters()).device)
        # print("模型总参数：", sum(p.numel() for p in model.parameters()))
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

        #config.num_epoch,每则运行几个epoch
        for epoch in range(config.num_epoch):
            train_loss = 0.0
            train_correct = 0
            model.train()
            for i, (target, token_index) in enumerate(train_data_loader):
                # target, token_index = target.to(device), token_index.to(device)
                optimizer.zero_grad()
                target = target.to(device)
                token_index = token_index.to(device)
                output = model(token_index)
                loss = loss_func(output, target)  # 输出是tensor类型
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                train_loss += loss.item() * token_index.size(0)
                predictions = torch.argmax(output.data, 1)  # 统计以行为单位的最大值，计为scores，预测记为predictions
                train_correct += (predictions == target).sum().item()

            train_loss = train_loss / len(train_index)
            train_acc = train_correct / len(train_index) * 100  # 转换成100%
            scheduler.step(train_loss)  # 改变优化器里面的学习率

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Training Acc {:.2f} ls{:.5f}".
                  format(epoch + 1, config.num_epoch, train_loss, train_acc,
                         optimizer.state_dict()['param_groups'][0]['lr']))

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # 输出训练曲线：
            writer.add_scalars('test_fold1', {'train_loss': train_loss, 'train_acc': train_acc}, epoch)

        writer.close()
            # 训练完一次就测试一次：
            # test_acc, f1, precision, recall = valid_epoch(test_data_loader, model, loss_func, device)


        foldperf['fold{}'.format(kfold + 1)] = history

        # 全部训练玩之后测试，之后进行F1计算
        test_acc, f1, precision, recall = valid_epoch(test_data_loader, model, loss_func, device)

        meanF1.append(f1)
        meanPre.append(precision)
        meanRe.append(recall)
        # 平均损失输出
        writer.add_scalars('test', {'train_loss': np.mean(history['train_loss']),
                                    'test_acc': test_acc, 'train_acc': np.mean(history['train_acc']),
                                    'F1': f1, 'Precision': precision, 'Recall': recall}, kfold+1)
        #保存模型
        torch.save(model.state_dict(), '../Kfold_model/'+str(kfold+1)+'_model.pth')


    writer.close()
    print('  平均F1 : ' + str(sum(meanF1)) +
          '  平均Precision : ' + str(sum(meanPre)) +
          '  平均Recall : ' + str(sum(meanRe)))









