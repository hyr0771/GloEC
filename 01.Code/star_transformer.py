import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from model_layer import Classifier

class Embedding(torch.nn.Module):
    def __init__(self, char_map, config, padding_idx):
        super(Embedding, self).__init__()
        self.dropout = torch.nn.Dropout(p=config.embedding_dropout)  # ？？？
        self.embedding = torch.nn.Embedding(
            len(char_map), config.embedding_dimension, padding_idx=padding_idx)
        if config.embedding_init:
            if config.embedding_init_type == 'uniform':            # 初始化为均匀分布
                embedding_lookup_table = torch.nn.init.uniform_(tensor=torch.empty(len(char_map),
                                        config.embedding_dimension), a=-0.5, b=0.5)  # (21*64)
                # a 和 b是均匀分布中随机生成数字的上下界限

            elif config.embedding_init_type == 'normal':      # 正态分布
                embedding_lookup_table = torch.nn.init.uniform_(tensor=torch.empty(len(char_map),
                                        config.embedding_dimension))  # (21*64)
            if padding_idx is not None:
                embedding_lookup_table[padding_idx] = 0.0  # 把pad符号的embedding设置为全0
            self.embedding.weight.data.copy_(embedding_lookup_table)

    def forward(self, vocab_ids):
        embedding = self.embedding(vocab_ids)
        return self.dropout(embedding)

class Transformer(torch.nn.Module):
    def __init__(self, config, char_map, label_map):
        super(Transformer, self).__init__()
        self.config = config

        self.char_embedding = Embedding(char_map, config, config.padID)

        self.pad = config.padID
        # 用char
        seq_max_len = config.feature_max_char_len
        # 创建位置编码矩阵
        self.position_enc = PositionEmbedding(seq_max_len,
                                              config.embedding_dimension,
                                              self.pad)

        self.layer_stack = nn.ModuleList([
            StarEncoderLayer(config.embedding_dimension,
                             config.Transformer.n_head,
                             config.Transformer.d_k,
                             config.Transformer.d_v,
                             dropout=config.Transformer.encoder_dropout)
            for _ in range(config.Transformer.n_layers)])


        hidden_size = config.embedding_dimension

        self.linear = torch.nn.Linear(hidden_size, 7)
        self.dropout = torch.nn.Dropout(p=config.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append({'params': self.char_embedding.parameters()})
        for i in range(0, len(self.layer_stack)):
            params.append({'params': self.layer_stack[i].parameters()})
        params.append({'params': self.linear.parameters()})
        return params


    def forward(self, batch, length=None):
        # 函数里定义的函数只能函数内调用
        def _get_non_pad_mask(seq, pad):
            assert seq.dim() == 2
            return seq.ne(pad).type(torch.float).unsqueeze(-1)  #

        # def _get_attn_key_pad_mask(seq_k, seq_q, pad):
        #     ''' For masking out the padding part of key sequence. '''
        #
        #     # Expand to fit the shape of key query attention matrix.
        #     len_q = seq_q.size(1)
        #     padding_mask = seq_k.eq(pad)
        #     padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
        #
        #     return padding_mask

        src_seq = batch.to(self.config.device)
        embedding = self.char_embedding(src_seq)

        # Prepare masks
        # slf_attn_mask = _get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, pad=self.pad)
        non_pad_mask = _get_non_pad_mask(src_seq, self.pad)

        batch_lens = (src_seq != self.pad).sum(dim=-1)
        src_pos = torch.zeros_like(src_seq, dtype=torch.long)
        for row, length in enumerate(batch_lens):
            src_pos[row][:length] = torch.arange(1, length + 1)

        enc_output = embedding + self.position_enc(src_pos)

        s = torch.mean(embedding, 1)  # virtual relay node
        h = enc_output
        for enc_layer in self.layer_stack:
            h, s = enc_layer(h, embedding, s,
                             non_pad_mask=non_pad_mask,  # 这里non_pad_mask = non_pad_mask，先改成None
                             slf_attn_mask=None)
        h_max, _ = torch.max(h, 1)
        enc_output = h_max + s

        return self.dropout(self.linear(enc_output))

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class StarEncoderLayer(nn.Module):
    ''' Star-Transformer: https://arxiv.org/pdf/1902.09113v2.pdf '''

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super(StarEncoderLayer, self).__init__()
        self.slf_attn_satellite = MultiHeadAttention(
            n_head, d_model, d_k, d_v, use_star=True, dropout=dropout)
        self.slf_attn_relay = MultiHeadAttention(
            n_head, d_model, d_k, d_v, use_star=True, dropout=dropout)

    def forward(self, h, e, s, non_pad_mask=None, slf_attn_mask=None):
        # satellite node
        batch_size, seq_len, d_model = h.size()
        h_extand = torch.zeros(batch_size, seq_len+2, d_model, dtype=torch.float, device=h.device)
        h_extand[:, 1:seq_len+1, :] = h  # head and tail padding(not cycle)
        s = s.reshape([batch_size, 1, d_model])
        s_expand = s.expand([batch_size, seq_len, d_model])
        context = torch.cat((h_extand[:, 0:seq_len, :],
                             h_extand[:, 1:seq_len+1, :],
                             h_extand[:, 2:seq_len+2, :],
                             e,
                             s_expand),
                            2)
        context = context.reshape([batch_size*seq_len, 5, d_model])
        h = h.reshape([batch_size*seq_len, 1, d_model])

        h, _ = self.slf_attn_satellite(
            h, context, context, mask=slf_attn_mask)
        h = torch.squeeze(h, 1).reshape([batch_size, seq_len, d_model])
        if non_pad_mask is not None:
            h *= non_pad_mask

        # virtual relay node
        s_h = torch.cat((s, h), 1)
        s, _ = self.slf_attn_relay(
            s, s_h, s_h, mask=slf_attn_mask)
        s = torch.squeeze(s, 1)

        return h, s

class PositionEmbedding(torch.nn.Module):
    ''' Reference: attention is all you need '''

    def __init__(self, seq_max_len, embedding_dim, padding_idx):
        super(PositionEmbedding, self).__init__()


        # 执行一手from_pretrained的目的是什么
        self.position_enc = nn.Embedding.from_pretrained(
            self.get_sinusoid_encoding_table(seq_max_len + 1,
                                             embedding_dim,
                                             padding_idx=padding_idx), freeze=True)

    def forward(self, src_pos):
        return self.position_enc(src_pos)

    # @staticmethod  # 可以类名.方法直接调用，不用创建实例类
    def get_sinusoid_encoding_table(self, n_position, d_hid, padding_idx=None):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array(
            [get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.

        return torch.FloatTensor(sinusoid_table)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, use_star=False, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.use_star = use_star

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        if self.use_star:
            output = self.dropout(F.relu(self.fc(output)))
            output = self.layer_norm(output)
        else:
            output = self.dropout(self.fc(output))
            output = self.layer_norm(output + residual)

        return output, attn