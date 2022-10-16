'''
@Author: WANG Maonan
@Date: 2022-09-27 08:21:38
@Description: 将注意力机制加入 Seq2Seq 模型
@LastEditTime: 2022-10-14 09:11:51
'''
import torch
from torch import nn
from torch.nn import functional as F

from lib.d2l_torch import Seq2SeqEncoder, Decoder, AdditiveAttention, init_seq2seq


class AttentionDecoder(Decoder):
    """The base attention-based decoder interface.
    """
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        """这里主要用于获得 attention weight, 来绘制图形
        """
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    """Encoder 部分是没有改变的, Decoder 部分加入了 Attention.
    """
    def __init__(
                self, 
                vocab_size, 
                embed_size, 
                num_hiddens, 
                num_layers,
                dropout=0
            ):
        super().__init__()
        self.attention = AdditiveAttention(num_hiddens, dropout) # 加性 Attention
        self.embedding = nn.Embedding(vocab_size, embed_size) # Embedding 层
        self.rnn = nn.GRU(
            embed_size + num_hiddens, 
            num_hiddens, 
            num_layers,
            dropout=dropout
        )
        self.dense = nn.LazyLinear(vocab_size) # 输出每一个词的概率
        self.apply(init_seq2seq)

    def init_state(self, enc_outputs, enc_valid_lens):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        # enc_valid_lens, 表示 encode 句子中哪些是 padding 的
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # 这里 num_steps 是原始句子的长度
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens). 每一的 output --> 作为 key 和 value
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens), final hidden state
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)

        # #####################
        # 下面用到了 Attention 的机制
        # #####################
        outputs, self._attention_weights = [], []
        for x in X: # 每一个字
            # Shape of query: (batch_size, 1, num_hiddens)
            # query 是上一个时间的 RNN 的输出, 其实就是上一个字 Embedding 的结果
            query = torch.unsqueeze(hidden_state[-1], dim=1) # 只取最后一个 layer 的结果
            # Shape of context: (batch_size, 1, num_hiddens)
            # Shape of enc_outputs: (batch size, num_steps, h)
            # query, key, value
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens) # context 是如何计算的
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1) # context 和 x 合并起来
            # Reshape x from (batch_size, 1, embed_size + num_hiddens) to (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights) # 存储 attention weights
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        print('-----')
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

if __name__ == '__main__':
    vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
    batch_size, num_steps = 4, 7 # num_steps 为句子的长度
    encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
    decoder = Seq2SeqAttentionDecoder(vocab_size, embed_size, num_hiddens, num_layers)
    X = torch.zeros((batch_size, num_steps), dtype=torch.long)
    state = decoder.init_state(encoder(X), None) # encoder(X) 返回 outputs, hidden_state
    output, state = decoder(X, state)
    print(output.shape, (batch_size, num_steps, vocab_size))
    print(state[0].shape, (batch_size, num_steps, num_hiddens))
    print(state[1][0].shape, (batch_size, num_hiddens))
