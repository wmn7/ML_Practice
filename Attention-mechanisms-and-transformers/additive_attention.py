'''
@Author: WANG Maonan
@Date: 2022-09-26 13:52:49
@Description: 加性 Attention 的介绍
@LastEditTime: 2022-09-26 14:09:40
'''
import torch
from torch import nn

from lib.d2l_torch import masked_softmax

class AdditiveAttention(nn.Module):
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False) # k --> h
        self.W_q = nn.LazyLinear(num_hiddens, bias=False) # q --> h
        self.w_v = nn.LazyLinear(1, bias=False) # h --> 1
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # 注意这里维度的变化, 会有四个维度
        # valid_lens, 考虑多少个 key-value pair
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, 
        # shape of queries: (batch_size, no. of queries, 1, num_hiddens)
        # shape of keys: (batch_size, 1, no. of key-value pairs, num_hiddens). 
        # Sum them up with broadcasting
        # 最终结果的维度是, (batch_size, no. of queries, no. of key-value pairs, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. 
        # Shape of scores: (batch_size, no. of queries, no. of key-value pairs)
        # 对每一个 query, 都有 key-value pair 的大小
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)

if __name__ == '__main__':
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # The two value matrices in the values minibatch are identical
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6]) # 第一个查看前 2 个, 第二个查看前 6 个

    attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
    attention.eval()
    attention_score = attention(queries, keys, values, valid_lens)
    print(attention_score)