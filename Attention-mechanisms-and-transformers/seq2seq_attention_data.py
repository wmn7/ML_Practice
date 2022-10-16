'''
@Author: WANG Maonan
@Date: 2022-10-14 10:43:42
@Description: 使用数据集来训练一个 Attention 的网络
@LastEditTime: 2022-10-15 12:08:51
'''
import torch
from torch import nn

from lib.d2l_torch import GRU, Encoder, Decoder, AdditiveAttention, init_seq2seq, MTFraEng, Seq2Seq, Trainer, bleu, try_gpu

# ########
# 定义模型
# ########
class Seq2SeqEncoder(Encoder):
    """The RNN encoder for sequence to sequence learning.

    Defined in :numref:`sec_seq2seq`"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = GRU(embed_size, num_hiddens, num_layers, dropout)
        self.apply(init_seq2seq)

    def forward(self, X, *args):
        # X shape: (batch_size, num_steps), 这里 num_steps 是句子的长度, 例如是 (128, 9)
        embs = self.embedding(X.t().type(torch.int64))
        # embs shape: (num_steps, batch_size, embed_size), 例如 torch.Size([9, 128, 256])
        # 这个 embs shape 是 GRU 规定的输入大小,  (L, N, H) when batch_first=False 
        output, state = self.rnn(embs)
        # output shape: (num_steps, batch_size, num_hiddens), torch.Size([9, 128, 256])
        # state shape: (num_layers, batch_size, num_hiddens), torch.Size([2, 128, 256])
        return output, state
        
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
        # Shape of outputs: (num_steps, batch_size, num_hiddens). 为 encoder 中的 output, 大小为, torch.Size([9, 128, 256])
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens), 初始化为 encoder 中的 hidden state, 大小为 torch.Size([2, 128, 256])
        # enc_valid_lens, 表示 encode 句子中哪些是 padding 的
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens) # output 大小转换后为 torch.Size([128, 9, 256])

    def forward(self, X, state):
        # 这里 num_steps 是原始句子的长度
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens). 每一的 output --> 作为 key 和 value, torch.Size([128, 9, 256])
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens), torch.Size([2, 128, 256])
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2) # torch.Size([9, 128, 256])

        # #########################
        # 下面用到了 Attention 的机制
        # #########################
        outputs, self._attention_weights = [], [] 
        for x in X: # decoder 中每一个字需要依次输入
            # Shape of query: (batch_size, 1, num_hiddens), torch.Size([128, 1, 256])
            # query 是上一个时间的 RNN 的输出, 其实就是上一个字 Embedding 的结果
            query = torch.unsqueeze(hidden_state[-1], dim=1) # 只取最后一个 layer 的结果
            # Shape of enc_outputs: (batch size, num_steps, h), torch.Size([128, 9, 256])
            # Shape of context: (batch_size, 1, num_hiddens), torch.Size([128, 1, 256]), 加权之后的结果
            # query, key, value
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens) # Attention 的关键, context 的计算
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1) # context 和 x 合并起来
            # 最终 x 的大小为, torch.Size([128, 1, 512])
            # Reshape x from (batch_size, 1, embed_size + num_hiddens) to (1, batch_size, embed_size + num_hiddens), 此时是为了适应 gru 输入的大小
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state) # 这里 out 的大小为 torch.Size([1, 128, 256])
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights) # 存储 attention weights

        outputs = self.dense(torch.cat(outputs, dim=0)) # cat 之后大小为 torch.Size([9, 128, 256])
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size), 也就是 torch.Size([9, 128, 214])
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

if __name__ == '__main__':
    data = MTFraEng(batch_size=128)
    embed_size, num_hiddens, num_layers, dropout = 256, 512, 2, 0.2

    encoder = Seq2SeqEncoder(
        vocab_size=len(data.src_vocab), # 194
        embed_size=embed_size, 
        num_hiddens=num_hiddens, 
        num_layers=num_layers, 
        dropout=dropout
    )
    decoder = Seq2SeqAttentionDecoder(
        vocab_size=len(data.tgt_vocab), # 214
        embed_size=embed_size, 
        num_hiddens=num_hiddens, 
        num_layers=num_layers, 
        dropout=dropout
    )
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'], lr=0.005)

    # 训练模型
    trainer = Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
    trainer.fit(model, data)

    # 测试模型
    engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    preds, _ = model.predict_step(
        data.build(engs, fras), try_gpu(), data.num_steps)
    for en, fr, p in zip(engs, fras, preds):
        translation = []
        for token in data.tgt_vocab.to_tokens(p):
            if token == '<eos>':
                break
            translation.append(token)
        print(f'{en} => {translation}, bleu,'
            f'{bleu(" ".join(translation), fr, k=2):.3f}')