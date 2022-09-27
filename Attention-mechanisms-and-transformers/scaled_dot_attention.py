'''
@Author: WANG Maonan
@Date: 2022-09-27 07:00:16
@Description: 内积注意力介绍
@LastEditTime: 2022-09-27 07:17:06
'''
import math
import torch
from torch import nn
from torch.nn import functional as F

from lib.d2l_torch import masked_softmax

class DotProductAttention(nn.Module):
    """Scaled dot product attention. (这里是没有参数的)
    """
    def __init__(self, dropout, num_heads=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads  # To be covered later

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None, window_mask=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        if window_mask is not None:  # To be covered later
            num_windows = window_mask.shape[0]
            n, num_queries, num_kv_pairs = scores.shape
            # Shape of window_mask: (num_windows, no. of queries,
            # no. of key-value pairs)
            scores = scores.reshape((n // (num_windows * self.num_heads), num_windows, self.
    num_heads, num_queries, num_kv_pairs)) + window_mask.unsqueeze(1).unsqueeze(0)
            scores = scores.reshape((n, num_queries, num_kv_pairs))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


if __name__ == '__main__':
    # 有 1 个 query, query 的长度是 2
    # 有 10 个 key, key 的长度是 2 (这里 key 和 query 的长度是一样的)
    # 有 10 个 value, value 的长度是 4
    queries, keys = torch.normal(0, 1, (2, 1, 2)), torch.ones((2, 10, 2))
    # The two value matrices in the values minibatch are identical
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6]) # 第一个查看前 2 个, 第二个查看前 6 个

    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))