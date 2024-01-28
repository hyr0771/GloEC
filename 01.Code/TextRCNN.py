import torch
import torch.nn.functional as F
# from model_layer import Classifier
import torch

class TextRCNN(torch.nn.Module):
    """TextRNN + TextCNN
    """
    def __init__(self, config, char_map, label_map):
        super(TextRCNN, self).__init__()
        self.rnn = RNN(
            config.embedding_dimension, config.TextRCNN.hidden_dimension,
            num_layers=config.TextRCNN.num_layers,
            batch_first=True, bidirectional=config.TextRCNN.bidirectional,
            rnn_type=config.TextRCNN.rnn_type)

        hidden_dimension = config.TextRCNN.hidden_dimension
        if config.TextRCNN.bidirectional:
            hidden_dimension *= 2
        self.kernel_sizes = config.TextRCNN.kernel_sizes
        self.convs = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(torch.nn.Conv1d(
                hidden_dimension, config.TextRCNN.num_kernels,
                kernel_size, padding=kernel_size - 1))

        self.top_k = self.config.TextRCNN.top_k_max_pooling
        hidden_size = len(config.TextRCNN.kernel_sizes) * \
                      config.TextRCNN.num_kernels * self.top_k

        if config.with_father_label:
            self.linear = torch.nn.Linear(hidden_size, len(label_map))
        else:
            # 计算纯属亚子类的类别有多少个
            len_num = 0
            for key, value in label_map.items():
                num = str(key).replace('\n', '').replace(' ', '')
                num_list = num.split('.')
                if len(num_list) == 3:
                    len_num += 1
            self.linear = torch.nn.Linear(hidden_size, len_num)

        self.dropout = torch.nn.Dropout(p=config.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append({'params': self.char_embedding.parameters()})  # 存在大问题，为什么Edbedding层的这个也要加进来进行优化？
        params.append({'params': self.rnn.parameters()})
        params.append({'params': self.convs.parameters()})
        params.append({'params': self.linear.parameters()})
        return params

    def forward(self, batch, length):
        embedding = self.char_embedding(
            batch.to(self.config.device))

            # batch[cDataset.DOC_CHAR_LEN] 每个句子的实际长度
        seq_length = length.to(self.config.device)

        output, _ = self.rnn(embedding, length)

        doc_embedding = output.transpose(1, 2)
        pooled_outputs = []
        for _, conv in enumerate(self.convs):
            convolution = F.relu(conv(doc_embedding))
            pooled = torch.topk(convolution, self.top_k)[0].view(
                convolution.size(0), -1)
            pooled_outputs.append(pooled)

        doc_embedding = torch.cat(pooled_outputs, 1)

        return self.dropout(self.linear(doc_embedding))


class RNN(torch.nn.Module):
    """
    One layer rnn.
    """
    def __init__(self, input_size, hidden_size, num_layers=1,
                 nonlinearity="tanh", bias=True, batch_first=False, dropout=0,
                 bidirectional=False, rnn_type='GRU'):
        super(RNN, self).__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        if rnn_type == 'LSTM':
            self.rnn = torch.nn.LSTM(
                input_size, hidden_size, num_layers=num_layers, bias=bias,
                batch_first=batch_first, dropout=dropout,
                bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = torch.nn.GRU(
                input_size, hidden_size, num_layers=num_layers, bias=bias,
                batch_first=batch_first, dropout=dropout,
                bidirectional=bidirectional)
        elif rnn_type == 'RNN':
            self.rnn = torch.nn.RNN(
                input_size, hidden_size, vnonlinearity=nonlinearity, bias=bias,
                batch_first=batch_first, dropout=dropout,
                bidirectional=bidirectional)


    def forward(self, inputs, seq_lengths=None, init_state=None,
                ori_state=False):
        """
        Args:
            inputs:
            seq_lengths:
            init_state:
            ori_state: If true, will return ori state generate by rnn. Else will
                       will return formatted state
        :return:
        """
        if seq_lengths is not None:
            seq_lengths = seq_lengths.int()
            sorted_seq_lengths, indices = torch.sort(seq_lengths,
                                                     descending=True)
            if self.batch_first:
                sorted_inputs = inputs[indices]
            else:
                sorted_inputs = inputs[:, indices]
            packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_inputs, sorted_seq_lengths, batch_first=self.batch_first)
            outputs, state = self.rnn(packed_inputs, init_state)
        else:
            outputs, state = self.rnn(inputs, init_state)

        if ori_state:
            return outputs, state
        if self.rnn_type == 'LSTM':
            state = state[0]
        if self.bidirectional:
            last_layers_hn = state[2 * (self.num_layers - 1):]
            last_layers_hn = torch.cat(
                (last_layers_hn[0], last_layers_hn[1]), 1)
        else:
            last_layers_hn = state[self.num_layers - 1:]
            last_layers_hn = last_layers_hn[0]

        _, revert_indices = torch.sort(indices, descending=False)
        last_layers_hn = last_layers_hn[revert_indices]
        pad_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=self.batch_first)
        if self.batch_first:
            pad_output = pad_output[revert_indices]
        else:
            pad_output = pad_output[:, revert_indices]
        return pad_output, last_layers_hn