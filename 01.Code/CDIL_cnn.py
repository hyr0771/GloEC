import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
import torch.nn.functional as F

class CDIL_Block(nn.Module):
    def __init__(self, c_in, c_out, ks, pad, dil):
        super(CDIL_Block, self).__init__()
        self.conv = weight_norm(nn.Conv1d(c_in, c_out, ks, padding=pad, dilation=dil, padding_mode='circular'))  # 1600个参数
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.normal_(0, 0.01)

        self.res = nn.Conv1d(c_in, c_out, kernel_size=(1,)) if c_in != c_out else None  # 1*1的卷积核用来改变通道数
        if self.res is not None:
            self.res.weight.data.normal_(0, 0.01)
            self.res.bias.data.normal_(0, 0.01)

        self.nonlinear = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        res = x if self.res is None else self.res(x)
        return self.nonlinear(out) + res
        # return out

class CDIL_ConvPart(nn.Module):
    def __init__(self, dim_in, hidden_channels, ks=3):
        super(CDIL_ConvPart, self).__init__()
        layers = []
        num_layer = len(hidden_channels)
        for i in range(num_layer):
            this_in = dim_in if i == 0 else hidden_channels[i - 1]
            this_out = hidden_channels[i]
            this_dilation = 2 ** i
            this_padding = int(this_dilation * (ks - 1) / 2)
            layers += [CDIL_Block(this_in, this_out, ks, this_padding, this_dilation)]
        self.conv_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_net(x)

class CDIL_CNN(nn.Module):
    def __init__(self, vocab_len, embed_dim, output_class, num_channels, kernel_size=3):
        super(CDIL_CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_len, embed_dim, padding_idx=0)
        self.conv = CDIL_ConvPart(embed_dim, num_channels, kernel_size)
        # 平均池化层也可以考虑一下
        # self.max_pool2d = nn.MaxPool2d(kernel_size=2)  # 对最后两维度进行池化操作，降维，其中stride默认时为kernel_size
        self.attention = selfAttention(2, 16, 16)
        # self.linear = nn.Linear(256*8, 256)
        # self.relu = nn.ReLU()
        # self.classifier_linear = nn.Linear(256, output_class)  # [-1]: 列表最后一项
        self.classifier = nn.Linear(num_channels[-1], output_class)  # [-1]: 列表最后一项


    def forward(self, x):
        x = self.embedding(x)
        x = self.attention(x)
        x = x.permute(0, 2, 1).to(dtype=torch.float)
        x = self.conv(x)  # x, y: num, channel(dim), length


        y = self.classifier(torch.mean(x, dim=2))
        # x = self.max_pool2d(x)  # [batch * 256 * 8]
        # # 拉直，输出
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)
        # # 加reul
        # x = self.relu(x)
        # y = self.classifier_linear(x)
        return y

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

        attention_probs = F.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)
        return context


if __name__ == '__main__':
    features = torch.rand((64, 512, 16))
    attention = selfAttention(2, 16, 20)
    result = attention.forward(features)
    print(result.shape)

