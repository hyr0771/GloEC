import torch
import os

class Config(object):
    """配置参数"""
    def __init__(self):
        # 训练信息
        self.model_name = 'ESM'                # 'TextRCNN' or 'Transformer'
        self.model_save_path = '../Save_model/'    # 模型保存路径

        # self.model_type = 'single'           # 可以选择 ‘single_hiera’ 或者 ‘mutil_hiera’
        self.creat_vocab_data_path = '../Data/uniport_2022_5/data.csv'    #文件路径
        self.train_data_path = '../Data/uniport_2022_5/f_train.csv'          #文件路径
        self.eval_data_path = '../Data/uniport_2022_5/f_eval.csv'                #文件路径
        self.test_data_path = '../Data/uniport_2022_5/f_time_test.csv'                #文件路径
        # self.hiera_json = '../Data/split_c0.9/hierar.json'                 #层级关系文件
        self.label_map_json = '../Data/uniport_2022_5/f_conti_label_map.json'              
        self.data_path = '../Data/uniport_2022_5/'
        self.predict_result_path = '../Data/predict_result/'
        self.weight = '../Data/uniport_2022_5/f_train_weight.json'

        # self.loss_type = 'CrossEntropyLoss'       # "CrossEntropyLoss"  or  ’BCEWithLogitsLoss‘

        # 公共参数
        self.layer = 4                             # 1 , 2, 3 # 必须设置，当前分类的层级,将影响类别数
        self.use_biovec = False
        self.optimizer_type = "Adam"                # "Adam" or "SGD"
        self.PAD = 'pad'                            # padding符号
        self.padID = 0                              # char_map中第几个是pad,应该填从0开始数的数字，共有21个，最后为pad，所以填20
        self.embed_dim = 16                         # 16      # 新加入变量  子向量维度
        self.seq_len = 512                          # 512    # 最长句子
        self.epoch = 160                            
        self.batch_size = 512                       # mini-batch大小 
        self.learning_rate = 0.001
        self.is_dacay = True                        # 是否应用衰减
        self.landa = 0.8                            # 指数衰减指数,越小刚开始下降的越快，越大刚开始下降的越慢
        self.label_length = [7, 67, 210, 1652]
        self.similarity_half_label_length = [7, 67, 210, 1652]
        self.kfold = 10                             
        self.sure_full_kfold = False                
        self.kfold_epoch = 40                      
        self.kfold_dataset_csv = '../Data/uniport_2022_5/f_train.csv'
        self.kfold_dataset_esm = '../Data/uniport_2022_5/esm_f_train.pt'
        self.kfold_label_map = '../Data/uniport_2022_5/f_conti_label_map.json'
        self.iskfold = True

        # 关于层级惩罚
        self.use_hierar_penalty = True             # 是否使用层级惩罚
        self.hierar_penalty = 1e-6                  # 层级惩罚项力度  默认值
        self.use_GCN = True
        self.gcn_layer = 3                          
        self.layer_4_weight_degree = 'balenced'     # 第四层平衡程度


        # 是否强制使用GPU训练？
        self.must_gpu = False
        self.is_continue_train = False              # 是否要加载之前训练好的模型参数来训练，还是新建一个模型来训练
        self.continue_train_num = '06131454'                # 如果选择继续则应该修改加载的模型序号
        self.device = 'cpu'
        # 在这里判断有没有gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # 设置只有0号的GPU是可见的,超算要求的使用规范
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            print('使用GPU训练')
        else:
            if self.must_gpu:
                assert torch.cuda.is_available()
            else:
                print('使用CPU训练')
                self.device = 'cpu'


class Transformer_Config():
    """配置参数"""
    def __init__(self):
        self.encoder_dropout = 0.0  # encoder里面的dropout
        self.n_head = 8  
        self.d_k = 32  
        self.d_v = 32  
        self.n_layers = 6


class CDIL_CNN_Config():
    """配置参数"""
    def __init__(self):
        self.HIDDEN_CHANNEL = [24, 32, 32, 16]
        # self.LAYER = 4
